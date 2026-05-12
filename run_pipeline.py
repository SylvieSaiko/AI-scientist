"""
AI Scientist — Master Pipeline

Usage:
    python run_pipeline.py                 # full pipeline (phases 1-4)
    python run_pipeline.py --phase 1       # ideation only
    python run_pipeline.py --phase 2       # experimentation only
    python run_pipeline.py --phase 3       # manuscript write-up only
    python run_pipeline.py --phase 4       # automated review only
    python run_pipeline.py --idea-index 2  # use 3rd idea from archive (skip generation)
"""

import os, sys, json, argparse, time
from datetime import datetime

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORK_DIR)


def phase1(n_ideas: int = 10) -> dict:
    print("\n" + "="*70)
    print("PHASE 1 — IDEATION")
    print("="*70)
    from phase1_ideation.idea_generator import generate_ideas, select_best_idea
    ideas = generate_ideas(n=n_ideas)
    best  = select_best_idea()
    print(f"\nSelected idea: {best['title']}")
    print(f"  Composite score: {best['composite_score']:.1f}")
    print(f"  PubMed hits: {best['pubmed_hits']}")
    return best


def phase2(idea: dict, max_nodes: int = 3) -> dict:
    print("\n" + "="*70)
    print("PHASE 2 — EXPERIMENTATION (Agentic Tree Search)")
    print("="*70)
    cache = os.path.join(WORK_DIR, "data", "dataset.csv")
    if not os.path.exists(cache):
        print("[Pipeline] Loading dataset (first run)...")
        from data_loader import load_data, preprocess
        df = load_data()
        df = preprocess(df)
        df.to_csv(cache, index=False, encoding="utf-8")
        print(f"[Pipeline] Cached: {df.shape[0]:,} rows × {df.shape[1]} cols")
    else:
        print(f"[Pipeline] Using cached data: {cache}")

    from phase2_experimentation.tree_search import AgenticTreeSearch
    search = AgenticTreeSearch(idea, max_nodes_per_stage=max_nodes)
    summary = search.run()
    return summary


def phase3(search_summary: dict) -> str:
    print("\n" + "="*70)
    print("PHASE 3 — MANUSCRIPT WRITE-UP")
    print("="*70)
    from phase3_writeup.manuscript_writer import generate_manuscript
    return generate_manuscript(search_summary)


def phase4(n_reviewers: int = 5) -> dict:
    print("\n" + "="*70)
    print("PHASE 4 — AUTOMATED PEER REVIEW")
    print("="*70)
    from phase4_review.automated_reviewer import review_manuscript
    return review_manuscript(n_reviewers=n_reviewers)


def save_checkpoint(name: str, data):
    path = os.path.join(WORK_DIR, "outputs", f"{name}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[Checkpoint] Saved {path}")


def load_checkpoint(name: str):
    path = os.path.join(WORK_DIR, "outputs", f"{name}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="AI Scientist Pipeline")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run only this phase (1-4). 0 = all phases.")
    parser.add_argument("--n-ideas", type=int, default=10)
    parser.add_argument("--n-nodes", type=int, default=3)
    parser.add_argument("--n-reviewers", type=int, default=5)
    parser.add_argument("--idea-index", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()
    print(f"\nAI Scientist — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Working directory: {WORK_DIR}")
    run_all = args.phase == 0

    if run_all or args.phase == 1:
        if args.idea_index is not None:
            archive_path = os.path.join(WORK_DIR, "phase1_ideation", "idea_archive.json")
            with open(archive_path) as f:
                archive = json.load(f)
            idea = archive[args.idea_index]
        else:
            idea = phase1(n_ideas=args.n_ideas)
        save_checkpoint("phase1_idea", idea)
    else:
        idea = load_checkpoint("phase1_idea")
        if idea is None:
            print("ERROR: No phase 1 checkpoint. Run phase 1 first.")
            sys.exit(1)

    if args.phase == 1:
        print(f"\nPhase 1 complete in {(time.time()-t0)/60:.1f} min")
        return

    if run_all or args.phase == 2:
        search_summary = phase2(idea, max_nodes=args.n_nodes)
        save_checkpoint("phase2_search", search_summary)
    else:
        search_summary = load_checkpoint("phase2_search")
        if search_summary is None:
            print("ERROR: No phase 2 checkpoint.")
            sys.exit(1)

    if args.phase == 2:
        print(f"\nPhase 2 complete in {(time.time()-t0)/60:.1f} min")
        return

    if run_all or args.phase == 3:
        phase3(search_summary)
    if args.phase == 3:
        print(f"\nPhase 3 complete in {(time.time()-t0)/60:.1f} min")
        return

    if run_all or args.phase == 4:
        review = phase4(n_reviewers=args.n_reviewers)
        save_checkpoint("phase4_review", review)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*70}\nPIPELINE COMPLETE — {elapsed:.1f} minutes\n{'='*70}")
    print(f"  Idea:       outputs/phase1_idea.json")
    print(f"  Manuscript: phase3_writeup/manuscript.md")
    print(f"  Review:     phase4_review/review_report.json")
    print(f"  Figures:    phase3_writeup/figures/")


if __name__ == "__main__":
    main()
