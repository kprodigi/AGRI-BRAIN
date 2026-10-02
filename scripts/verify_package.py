"""Verify the distributed files and evidence without running simulations."""
from pathlib import Path
import hashlib,json,csv,math,sys,ast,re
from collections import Counter
R=Path(__file__).resolve().parents[1]
def load(p): return json.loads((R/p).read_text(encoding="utf-8"))
def rows(p):
    with (R/p).open(encoding="utf-8",newline="") as f: return list(csv.DictReader(f))
def check(ok,message):
    if not ok: raise AssertionError(message)
def check_links():
    """Every relative link and image in the README and the documentation pages must exist."""
    pages=[R/"README.md",*sorted((R/"docs").glob("*.md")),R/"figures"/"README.md"]
    pattern=re.compile(r'\]\(([^)\s]+)\)|src="([^"]+)"')
    for page in pages:
        for markdown,html in pattern.findall(page.read_text(encoding="utf-8")):
            target=(markdown or html).split("#")[0]
            if not target or target.startswith(("http://","https://","mailto:")): continue
            check((page.parent/target).exists(),"Broken link in "+page.relative_to(R).as_posix()+": "+target)
def main():
    manifest=load("FILE_HASHES.json")
    for name,digest in manifest.items():
        check(hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,"Checksum mismatch: "+name)
    modes={"static","no_context","agribrain"}
    primary=rows("data/primary/three_mode_endpoints.csv")
    index={(x["seed"],x["scenario"],x["mode"]):x for x in primary}
    check(len(index)==len(primary)==300,"Primary coverage")
    check(set(x["mode"] for x in primary)==modes,"Unexpected primary mode")
    check(len(set(x["seed"] for x in primary))==20,"Seed coverage")
    check(len(set(x["scenario"] for x in primary))==5,"Scenario coverage")
    paired=rows("data/primary/paired_seed_ARI.csv")
    check(len(paired)==100,"Paired coverage")
    for x in paired:
        a=float(index[x["seed"],x["scenario"],"agribrain"]["ari"])
        b=float(index[x["seed"],x["scenario"],"no_context"]["ari"])
        check(abs(a-float(x["agribrain"]))<1e-12 and abs(b-float(x["without_context"]))<1e-12,"Paired endpoints")
        check(abs(a-b-float(x["gain"]))<1e-12,"Paired difference")
    sensitivity=rows("data/sensitivity/seed_endpoints.csv")
    check(len(sensitivity)==6900,"Sensitivity coverage")
    check(len({(x["setting"],x["seed"],x["scenario"],x["mode"]) for x in sensitivity})==6900,"Duplicate sensitivity cell")
    check(set(x["mode"] for x in sensitivity)==modes,"Unexpected sensitivity mode")
    check(set(Counter(x["setting"] for x in sensitivity).values())=={300},"Setting coverage")
    nominal_max=0.0
    for x in sensitivity:
        if x["setting"]=="nominal":
            y=index[x["seed"],x["scenario"],x["mode"]]
            for k in ("ari","waste","rle","slca","carbon","equity"):
                nominal_max=max(nominal_max,abs(float(x[k])-float(y[k])))
    check(nominal_max<1e-9,"Nominal primary/sensitivity discrepancy")
    sys.path.insert(0,str(R/"reproduction/source/agribrain/backend"))
    from src.models.mode_capabilities import VALID_MODES,capabilities_for
    check(set(VALID_MODES)==modes,"Public mode registry")
    for obsolete in ("mcp_only","pirag_only","hybrid_rl","no_pinn","no_slca","agribrain_standard_rag"):
        try: capabilities_for(obsolete)
        except ValueError: pass
        else: raise AssertionError("Obsolete mode accepted: "+obsolete)
    for p in R.rglob("*.py"): ast.parse(p.read_text(encoding="utf-8-sig"),filename=str(p))
    figures=load("figures/figure_index.json")
    check(len(figures)==6 and len({x["file"] for x in figures})==6,"Figure coverage")
    check(all((R/"figures"/x["file"]).is_file() for x in figures),"Missing figure")
    check(all((R/d).is_file() for x in figures for d in x["data"]),"Missing figure data")
    check((R/"docs/images/architecture.jpg").is_file(),"Missing architecture diagram")
    check_links()
    gain=sum(float(x["gain"]) for x in paired)/len(paired)
    print(json.dumps({"status":"PASS","files_verified":len(manifest),"primary_evaluations":300,"sensitivity_evaluations":6900,"figures":6,"overall_paired_ari_gain":gain,"maximum_nominal_endpoint_difference":nominal_max,"simulation_rerun":False},indent=2))
if __name__=="__main__":main()
