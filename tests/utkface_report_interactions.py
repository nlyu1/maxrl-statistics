import argparse
import csv
import json
import re
from pathlib import Path

import quickjs

DOM_HARNESS = r"""
const nodes = new Map(), created = [];
class Element {
 constructor(tag='div'){this.tagName=tag;this.children=[];this._text='';this.attrs={};this.listeners={};this.style={setProperty(){}};this.dataset={};this.value='';this.checked=false;created.push(this);}
 get textContent(){return this._text+this.children.map(x=>x.textContent||'').join('');}
 set textContent(v){this._text=String(v);this.children=[];}
 append(...items){this.children.push(...items);}
 replaceChildren(...items){this.children=[];this._text='';this.append(...items);}
 setAttribute(k,v){this.attrs[k]=String(v);if(k==='data-view')this.dataset.view=v;}
 getAttribute(k){return this.attrs[k];}
 addEventListener(k,f){(this.listeners[k]||(this.listeners[k]=[])).push(f);}
 click(){(this.listeners.click||[]).forEach(f=>f({}));}
 querySelectorAll(){return this.children;}
 getBoundingClientRect(){return {width:600,height:300,left:0,top:0};}
 scrollIntoView(){}
 get selectedOptions(){const names={epoch:'Epoch',updates:'Optimizer updates',examples:'Training presentations',rmse_mean:'Mean-prediction RMSE'};return [{textContent:names[this.value]||this.value}];}
}
const document={getElementById(id){if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);},createElement:t=>new Element(t),createElementNS:(ns,t)=>new Element(t),documentElement:new Element()};
document.getElementById('report-data').textContent=snapshot_json;
for(const [id,value] of Object.entries({'size':'all','epochs':'all','start-epoch':'1','noise-mode':'fixed','noise-std':'5','noise-field':'2026','x-axis':'epoch','extra-metric':'rmse_mean','tolerance':'0.1'}))document.getElementById(id).value=value;
document.getElementById('seed-sd').checked=true;
for(const name of ['main','subset','schedule','scalar','noise','digits']){const b=new Element('button');b.dataset.view=name;document.getElementById('navigation').append(b);}
const window={innerWidth:1280,innerHeight:900,matchMedia:()=>({matches:false,addEventListener(){}})};
class ResizeObserver{constructor(f){}observe(){}}
const setTimeout=()=>1,clearTimeout=()=>{};
let downloaded='';
class Blob{constructor(parts){downloaded=parts.join('');}}
const URL={createObjectURL:()=>'',revokeObjectURL:()=>{}};
"""

ASSERTIONS = r"""
function check(condition,message){if(!condition)throw Error(message);}
check(state.groups.length===24,'full-data view must contain 22 categorical methods and two scalar heads');
check(state.groups.filter(g=>g.r.head==='categorical').length===22&&state.groups.filter(g=>g.r.head==='scalar').length===2,'full-data head coverage');
check(state.groups.reduce((n,g)=>n+g.n,0)===D.runs.filter(r=>['main','scalar'].includes(r.stage)&&r.status==='completed').length,'full-data coverage mismatch');
check(D.summary.completed===D.runs.filter(r=>r.status==='completed').length,'completed study total');
check(D.summary.evaluations===D.runs.filter(r=>['main','scalar','subset','subset_replication','matched_steps','flat','noise','noise_replication','digits'].includes(r.stage)&&r.status==='completed').length,'scientific evaluation total excludes tuning and preflight');
document.getElementById('hide-all').click();check(state.groups.length===0,'hide all');
document.getElementById('show-all').click();check(state.groups.length===24,'show all');
document.getElementById('baselines').click();check(state.groups.length===5,'baseline filter includes scalar heads');
document.getElementById('show-all').click();
document.getElementById('start-epoch').value='20';render();
check(state.groups.every(g=>curve(g,1).length>0&&curve(g,1).every(p=>p.x>=20)),'late-epoch curve range');
document.getElementById('start-epoch').value='1';render();
const headers=document.getElementById('results-table').children[0].children[0].children[0].children;
headers[4].children[0].click();check(!state.ascending,'descending sort');
check(document.getElementById('results-table').children[0].children[1].children[0].children[0].textContent.startsWith(label(sorted(state.groups)[0].r)),'sorted first row');
details(state.groups[0]);check(!document.getElementById('run-details').hidden,'run evidence');
exportRows();check(downloaded.includes('selected_test_mae'),'CSV export');
for(const b of document.getElementById('navigation').children){b.click();check(state.view===b.dataset.view,'section navigation');}
view('subset');
check(document.getElementById('size').value==='4096'&&!document.getElementById('matched').checked,'subset defaults retain all available seeds');
check(state.groups.filter(g=>g.r.head==='categorical').length===8&&state.groups.filter(g=>g.r.head==='categorical').every(g=>g.n===3),'all eight categorical subset replications visible');
check(state.groups.filter(g=>g.r.head==='scalar').every(g=>g.n===1),'scalar subset seed count remains one');
document.getElementById('size').value='1024';document.getElementById('epochs').value='200';render();
check(state.groups.length===8&&state.groups.every(g=>g.r.epochs===200&&g.n===1),'all eight matched-update runs visible');
document.getElementById('epochs').value='50';render();
check(state.groups.length===10&&state.groups.every(g=>g.r.epochs===50&&g.n===1),'1024-image 50-epoch comparisons visible');
document.getElementById('partial').checked=true;render();
view('noise');for(const id of ['noise-mode','noise-std','noise-field'])document.getElementById(id).value='all';render();
check(state.groups.filter(g=>g.r.noise!=='none').length===80,'all eighty noise settings visible');
check(filtered().every(r=>r.status!=='held'),'held repetitions do not inflate comparison coverage');
view('digits');
check(state.groups.filter(g=>g.r.head==='digits').length===8,'all eight 4096-image digit objectives visible');
document.getElementById('size').value='18462';render();
check(state.groups.filter(g=>g.r.head==='digits').length===8,'all eight full-data digit objectives visible');
check(state.groups.filter(g=>g.r.method.startsWith('stat_maxrl')).every(g=>g.r.width===2),'digit baselines use matching statistical kernel widths');
view('main');document.getElementById('extra-metric').value='mode_accuracy';render();
check(document.getElementById('results-table').textContent.includes('mode_accuracy'),'additional metric');
check(created.filter(n=>n.tagName==='path').every(n=>!/(NaN|Infinity|undefined)/.test(n.attrs.d||'')),'nonfinite chart geometry');
check(created.filter(n=>n.tagName==='svg').length>0,'charts missing');
const hit=created.find(n=>n.tagName==='rect'&&n.listeners.pointermove);
hit.listeners.pointermove[0]({clientX:150,clientY:120});
check(!document.getElementById('tooltip').hidden,'curve hover');
hit.listeners.pointerleave[0]();check(document.getElementById('tooltip').hidden,'hover dismissal');
const baseline=D.runs.filter(r=>r.stage==='main'&&r.method==='mae'&&r.status==='completed');
const scalar={...baseline[0],id:'fixture-scalar',head:'scalar',seed:42};
document.getElementById('matched').checked=true;
const paired=groups([...baseline,scalar]);
const categorical=paired.find(g=>g.r.head==='categorical');
check(categorical.completedN===3&&categorical.n===1&&categorical.done[0].seed===42,'retain coverage while matching seeds');
check(categorical.values['test.mae_mean']===baseline.find(r=>r.seed===42).test.mae_mean,'paired summary uses matched baseline');
const pending=groups([...baseline,{...scalar,status:'running'}]).find(g=>g.r.head==='scalar');
const previousPartial=document.getElementById('partial').checked;
document.getElementById('partial').checked=true;
check(pending.n===0&&curve(pending,1).length>0,'unfinished curves can precede completed results');
document.getElementById('partial').checked=previousPartial;
document.getElementById('matched').checked=false;
"""


def verify(*, html_path: Path) -> dict:
    html = html_path.read_text()
    data = re.search(
        r'<script id="report-data" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    ).group(1)
    script = re.search(
        r'<script>\s*("use strict";.*?)</script>', html, re.DOTALL
    ).group(1)
    context = quickjs.Context()
    context.set("snapshot_json", data)
    context.eval(DOM_HARNESS)
    context.eval(script)
    context.eval(ASSERTIONS)
    exported = list(csv.reader(context.eval("downloaded").splitlines()))
    assert len(exported) == 25 and all(len(row) == 12 for row in exported)
    payload = json.loads(data)
    return {
        "snapshot": payload["generated"],
        "data_runs": len(payload["runs"]),
        "categorical_main_methods": 22,
        "full_data_methods": 24,
        "completed_jobs": payload["summary"]["completed"],
        "scientific_final_evaluations": payload["summary"]["evaluations"],
        "checks": [
            "script evaluation",
            "six sections",
            "hide/show/baseline controls",
            "descending first-row order",
            "per-run evidence",
            "CSV export and quoting",
            "subset and noise filters",
            "unfinished curves",
            "additional metric",
            "finite SVG geometry",
            "curve hover and dismissal",
            "common-seed comparisons preserve actual coverage",
            "unfinished curves before the first completed seed",
            "full-data categorical and scalar comparisons",
            "completed study counts exclude tuning and preflight evaluations",
            "late-epoch curve range preserves table results",
            "4096-image defaults retain all three categorical seeds",
            "1024-image 50/200-epoch horizon selection",
            "all eighty noise settings with held repetitions excluded",
            "all sixteen digit-policy objectives and matching baselines",
        ],
        "passed": True,
        "environment": "isolated QuickJS DOM harness; no browser rendering",
        "browser_preview_limitation": "In-app browser rejects file URLs; visual rendering not browser-verified",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = verify(html_path=args.html)
    args.output.write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
