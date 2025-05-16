import argparse
import csv
import json
import torch
import pytrec_eval
from tqdm import tqdm
from transformers import AutoTokenizer
from model import EnergyBasedDenseIR


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            qid, _, pid, rel = line.rstrip("\n").split("\t")
            if int(rel) > 0:
                qrels.setdefault(qid, {})[pid] = int(rel)
    return qrels


def load_query_passage_pairs(path):
    """Load pid qid query_text passage_text with arbitrary whitespace separators."""
    pairs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            # split on *first* three whitespace runs → preserve full passage
            parts = line.rstrip("\n").split(maxsplit=3)
            if len(parts) < 4:
                continue
            pid, qid, qtext, ptext = parts[0], parts[1], parts[2], parts[3]
            pairs.setdefault(qid, []).append((pid, qtext, ptext))
    return pairs


def evaluate(model, tokenizer, pairs, device, max_q_len=64, max_p_len=128):
    model.eval()
    run = {}
    with torch.no_grad():
        for qid in tqdm(pairs, desc="Scoring pairs"):
            run[qid] = {}
            for pid, qtext, ptext in pairs[qid]:
                q_tok = tokenizer(qtext, return_tensors="pt", truncation=True, max_length=max_q_len,
                                  padding="max_length").to(device)
                p_tok = tokenizer(ptext, return_tensors="pt", truncation=True, max_length=max_p_len,
                                  padding="max_length").to(device)
                energy = model(q_tok, p_tok)
                run[qid][pid] = -float(energy.item())  # lower energy ⇒ higher score
    return run

############################################
# Main
############################################

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model & tokenizer ---------------------------------------------------
    model = EnergyBasedDenseIR.load_from_checkpoint(args.ckpt_path)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    # data ----------------------------------------------------------------
    qrels = load_qrels(args.qrels_path)
    pairs = load_query_passage_pairs(args.pairs_path)

        # optional query limit -----------------------------------------------
    if args.limit_queries > 0:
        # keep the *first N queries that actually have qrels*
        overlap_qids = [qid for qid in pairs if qid in qrels]
        query_subset = overlap_qids[: args.limit_queries]
        pairs = {qid: pairs[qid] for qid in query_subset}
        qrels = {qid: qrels[qid] for qid in query_subset}

    overlap = set(pairs.keys()) & set(qrels.keys())
    print(f"# queries in candidate set : {len(pairs):,}")
    print(f"# queries with qrels       : {len(qrels):,}")
    print(f"# overlapping queries      : {len(overlap):,}\n")
    if len(overlap) == 0:
        print("[Error] No shared query IDs between candidates and qrels. Check you are using top1000.dev.tsv with qrels.dev.small.tsv.")
        return

    # score ---------------------------------------------------------------
    run = evaluate(model, tokenizer, pairs, device,
                   max_q_len=args.max_query_len, max_p_len=args.max_passage_len)

    # evaluate ------------------------------------------------------------
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank", "ndcg_cut.10", "recall_100"})
    results = evaluator.evaluate(run)
    if not results:
        print("[Error] pytrec_eval returned empty results likely no overlap between run and qrels.")
        return

    metrics = {m: sum(r[m] for r in results.values()) / len(results) for m in results[next(iter(results))]}
    print(json.dumps(metrics, indent=2))

############################################
# CLI
############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--encoder", default="bert-base-uncased")
    parser.add_argument("--pairs_path", required=True,
                        help="TSV with pid, qid, query text, passage text (top1000.dev.tsv)")
    parser.add_argument("--qrels_path", required=True,
                        help="qrels.dev.small.tsv")
    parser.add_argument("--limit_queries", type=int, default=100,
                        help="Evaluate first N queries (use -1 for all)")
    parser.add_argument("--max_query_len", type=int, default=64)
    parser.add_argument("--max_passage_len", type=int, default=128)
    main(parser.parse_args())
