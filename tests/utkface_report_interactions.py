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
for(const [id,value] of Object.entries({'size':'all','noise-mode':'fixed','noise-std':'5','noise-field':'2026','x-axis':'epoch','extra-metric':'rmse_mean','tolerance':'0.1'}))document.getElementById(id).value=value;
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
check(state.groups.length===22,'main matrix must contain all 22 methods');
check(state.groups.reduce((n,g)=>n+g.n,0)===D.runs.filter(r=>r.stage==='main'&&r.status==='completed').length,'main coverage mismatch');
document.getElementById('hide-all').click();check(state.groups.length===0,'hide all');
document.getElementById('show-all').click();check(state.groups.length===22,'show all');
document.getElementById('baselines').click();check(state.groups.length===3,'baseline filter');
document.getElementById('show-all').click();
const headers=document.getElementById('results-table').children[0].children[0].children[0].children;
headers[4].children[0].click();check(!state.ascending,'descending sort');
check(document.getElementById('results-table').children[0].children[1].children[0].children[0].textContent.startsWith(label(sorted(state.groups)[0].r)),'sorted first row');
details(state.groups[0]);check(!document.getElementById('run-details').hidden,'run evidence');
exportRows();check(downloaded.includes('selected_test_mae'),'CSV export');
for(const b of document.getElementById('navigation').children){b.click();check(state.view===b.dataset.view,'section navigation');}
view('subset');document.getElementById('size').value='1024';render();
document.getElementById('partial').checked=true;render();
view('noise');for(const id of ['noise-mode','noise-std','noise-field'])document.getElementById(id).value='all';render();
view('main');document.getElementById('extra-metric').value='mode_accuracy';render();
check(document.getElementById('results-table').textContent.includes('mode_accuracy'),'additional metric');
check(created.filter(n=>n.tagName==='path').every(n=>!/(NaN|Infinity|undefined)/.test(n.attrs.d||'')),'nonfinite chart geometry');
check(created.filter(n=>n.tagName==='svg').length>0,'charts missing');
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
    assert len(exported) == 23 and all(len(row) == 10 for row in exported)
    payload = json.loads(data)
    return {
        "snapshot": payload["generated"],
        "data_runs": len(payload["runs"]),
        "main_methods": 22,
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
