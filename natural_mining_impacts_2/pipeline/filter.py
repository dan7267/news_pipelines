import csv
import re
from pathlib import Path
from urllib.parse import urlparse, unquote
BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "interim" / "gdelt_event_context_daily"


# Tune these based on what you see in your URLs
NEGATIVE_PATH_KEYWORDS = [
    # --- SPORTS ---
    "sport","sports","football","soccer","nba","nfl","mlb","nhl","mma","ufc",
    "boxing","wrestling","tennis","golf","cricket","rugby","f1","formula-1",
    "motorsport","nascar","cycling","olympics","athletics","baseball",
    "basketball","hockey","esports","gaming",

    # --- ENTERTAINMENT ---
    "entertainment","celebrity","celebrities","hollywood","bollywood",
    "movies","movie","film","tv","television",
    "streaming","netflix","hulu","prime-video","amazon-prime","disney",
    "disney-plus","hbomax","spotify","music","album","song","songs",
    "concert","tour","festival","theatre","theater","broadway","oscars",
    "emmys","grammys","kardashian","royal-family","celeb",  # <-- FIXED COMMA

    # --- LIFESTYLE / POP CULTURE ---
    "lifestyle","fashion","beauty","makeup","skincare","hair","diet",
    "fitness","yoga","workout","gym","weightloss","wellness",
    "relationships","dating","wedding","weddings","sex","parenting",
    "horoscope","astrology","zodiac","tarot",

    # --- FOOD / RECIPES ---
    "recipe","recipes","cooking","cook","baking","kitchen","restaurant",
    "food","cuisine","dining","mayo","mayonnaise","chocolate","cake",
    "dessert","wine","beer","cocktail","coffee","tea",

    # --- TRAVEL / TOURISM ---
    "holiday","holidays","vacation","tourism","hotel","hotels",
    "cruise","beach","airport-guide",

    # --- TECH / GADGET REVIEWS ---
    "gadget","gadgets","smartphone","iphone","android",
    "laptop","tablet","camera","headphones","earbuds","tv-review",
    "gaming-console","ps5","xbox","nintendo",

    # --- GENERAL CLICKBAIT ---
    "quiz","giveaway","contest","sweepstakes","lottery",
    "viral","meme","memes","funny","top-10","top10",
    "slideshow","gallery","photos","pictures","wallpaper",

    # --- LOCAL HUMAN INTEREST / MISC ---
    "obituary","obituaries","funeral","wedding-announcement","birth",
    "anniversary","missing-person","pet","pets","dog","dogs","cat","cats","museum",
    "baby","babies","season","anime","laundry","ransom","scream","covid","equaliser",
    "hat-trick","school-shooting","homicide","candy","drugs","bicycle","burger",
    "methamphetamine","swimming","hiking","pizza","mcdonalds","magazine","turle",
    "elephant","flower","balloon","cinema","monster","cult","meth","demon","showbiz",
    "bishop","legend","moon","queer","roma","jewelery","mom","dad","horse","geforce",
    "sickle","casino","love","crypto","seals","honda","gender",
    "israel","gaza","palestine","israeli","genocide","child","children","schools",
    "epstein","university","palestinian","woman","arts","fraud","son","daughter",
    "art","wife","husband","gay","rape","parents","tvshowbiz","father","stabbing",
    "abortion","suicide","bitcoin","birthday","somalia","stabbed","cannabis","heroin",
    "biography","childcare","motel","diabetes","paedophile","virgin","graduates",
    "measles","baptist","mortgage","carjacking","motorcycle","jewelry","salmon",
    "afcon","robbers","vaccination","airbnb","dementia","spiritual","holocaust",
    "pornography","chelsea","chatgpt","childhood","cryptocurrency","opioid",
    "kidnap","pregnancy","raping","jews","raped","racism","priests","bishops",
    "somali","porn","boyfriend","girlfriend","somaliland","tiktok","fentanyl",
    "marijuana","antisemitism","teachers","podcast","jewish","trafficking",
    "migrants","oyster","church","mosque","synagogue", "cocaine", "drug"
]

# skip obvious non-article pages
NEGATIVE_PATH_PATTERNS = [
    r"/tag/", r"/tags/", r"/category/", r"/author/", r"/gallery/", r"/video/", r"/podcast/"
]

BAD_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp",
    ".mp4", ".mov", ".avi", ".zip", ".rar", ".7z"
)


def _url_search_text(p) -> str:
    """
    Build a normalized search string from parts of the URL so keywords match more reliably.
    Includes netloc + path + query. Decodes %xx and normalizes separators.
    """
    netloc = (p.netloc or "")
    path = unquote(p.path or "")
    query = unquote(p.query or "")

    full = f"{netloc} {path} {query}".lower()
    # normalize separators into spaces
    full = re.sub(r"[-_/]+", " ", full)
    # collapse whitespace
    full = re.sub(r"\s+", " ", full).strip()
    return full


def is_irrelevant_url(url: str) -> tuple[bool, str]:
    """
    Conservative URL-only filter.
    Returns (True, reason) if we should drop it.
    """
    url = (url or "").strip()
    if not url.startswith("http"):
        return True, "non_http"

    try:
        p = urlparse(url)

        # Extension check based on decoded path
        path_lower = unquote(p.path or "").lower()
        if path_lower.endswith(BAD_EXTENSIONS):
            return True, "bad_extension"

        # obvious section/category pages
        for pat in NEGATIVE_PATH_PATTERNS:
            if re.search(pat, (p.path or "").lower()):
                return True, f"neg_pattern:{pat}"

        # keyword match (netloc + path + query), with token + substring fallback
        full = _url_search_text(p)
        tokens = set(re.split(r"[^a-z0-9]+", full))

        for kw in NEGATIVE_PATH_KEYWORDS:
            kw_l = kw.lower().strip()
            if not kw_l:
                continue

            # token-level match (best for single words)
            if kw_l in tokens:
                return True, f"neg_kw_token:{kw_l}"

            # substring match (best for multiword/with hyphens like 'formula-1')
            if kw_l in full:
                return True, f"neg_kw_substr:{kw_l}"

        return False, ""
    except Exception:
        # if parsing fails, keep it (conservative)
        return False, ""


def dedupe_and_filter_file(path: Path) -> None:
    deduped_path = path.with_name(path.stem + "_deduped.csv")
    filtered_path = path.with_name(path.stem + "_deduped_filtered.csv")

    seen = set()
    kept_deduped = 0
    dropped_dupes = 0

    kept_filtered = 0
    dropped_irrelevant = 0

    with open(path, "r", newline="", encoding="utf-8") as f_in, \
         open(deduped_path, "w", newline="", encoding="utf-8") as f_deduped, \
         open(filtered_path, "w", newline="", encoding="utf-8") as f_filtered:

        reader = csv.reader(f_in)
        w_deduped = csv.writer(f_deduped)
        w_filtered = csv.writer(f_filtered)

        header = next(reader, None)
        if header is None:
            return

        w_deduped.writerow(header)
        w_filtered.writerow(header)

        try:
            url_idx = header.index("sourceurl")
        except ValueError:
            print(f"WARNING: no 'sourceurl' column in {path}")
            return

        for row in reader:
            if not row or len(row) <= url_idx:
                continue

            url = (row[url_idx] or "").strip()
            if not url:
                continue

            # 1) dedupe by URL
            if url in seen:
                dropped_dupes += 1
                continue

            seen.add(url)
            w_deduped.writerow(row)
            kept_deduped += 1

            # 2) filter irrelevant by URL parts
            drop, _reason = is_irrelevant_url(url)
            if drop:
                dropped_irrelevant += 1
                continue

            w_filtered.writerow(row)
            kept_filtered += 1

    print(
        f"{path.name}: "
        f"deduped kept {kept_deduped:,}, dupes dropped {dropped_dupes:,} | "
        f"filtered kept {kept_filtered:,}, irrelevant dropped {dropped_irrelevant:,}"
    )
    print(f"  -> {deduped_path.name}")
    print(f"  -> {filtered_path.name}")



def main(target_date: str): 
    # 1. We use rglob to search recursively through all subfolders (Year/Month/Day)
    # We look specifically for the raw context file for that date
    search_pattern = f"**/{target_date}_event_context.csv"
    files = list(BASE_DIR.rglob(search_pattern))
    
    if not files:
        # Fallback: search for any context file and filter manually by name 
        # (useful if the file naming convention varies slightly)
        all_context_files = list(BASE_DIR.rglob("*_event_context.csv"))
        files = [f for f in all_context_files if target_date in f.name]

    if not files:
        print(f"No files found for date {target_date} in {BASE_DIR}")
        return

    print(f"Found {len(files)} file(s) for {target_date}. Starting filtering...")

    for path in sorted(files):
        # The dedupe_and_filter_file function already uses path.with_name()
        # which means it will save the new CSVs in the EXACT same folder as the input.
        dedupe_and_filter_file(path)

if __name__ == "__main__":
    # If run by itself, ask for input
    day = input("Select date (YYYYMMDD): ").strip()
    main(day)