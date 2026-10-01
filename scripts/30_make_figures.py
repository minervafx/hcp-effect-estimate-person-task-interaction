"""Manuscript Figures 1-3 (SVG) from the JSON outputs of scripts 10-12; no estimation.

[release] Adapted from the manuscript build script: reads $HCPEE_OUT instead of the
project results tree; the three historical z-statistic bars of Figure 1 come from
expected/historical_zstat_reference.json (labelled historical). Drawing code unchanged.
Captions live in the manuscript and are not regenerated here.

[release v1.0.1] The Figure 3 footnote (Idiff attenuation and attribution p) is computed
from 01_anatomy_interaction.json instead of being written as fixed text.
"""
import html
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "hcpee"))
import paths                                                          # noqa: E402
O = Path(paths.OUT)
e=json.loads((O/'01_interaction_effect_estimate.json').read_text())
a=json.loads((O/'01_anatomy_interaction.json').read_text())
z=json.loads((Path(paths.REPO)/'expected'/'historical_zstat_reference.json').read_text())
r=json.loads((O/'02_rest_control.json').read_text())
FIG=O/'figures'
FIG.mkdir(parents=True, exist_ok=True)
COL=['#596779','#176b80','#a65936','#59764a']
def txt(x,y,s,size=12,color='#172b3a',anchor='start',bold=False):return f'<text x="{x}" y="{y}" font-family="Helvetica" font-size="{size}" fill="{color}" text-anchor="{anchor}"'+(' font-weight="bold"' if bold else '')+'>'+html.escape(str(s))+'</text>'
def chart(name,title,subtitle,panels,foot):
 W,H=680,365;out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="680" height="365" fill="white"/>',txt(15,26,title,17,bold=True),txt(15,48,subtitle,11)]
 for j,(lab,values,names,maxv,baseline) in enumerate(panels):
  left=25+j*220;bottom=252;plotw=184;ph=157
  out.append(txt(left,78,lab,13,bold=True))
  for v in [0,maxv/2,maxv]:
   y=bottom-ph*v/maxv;out.append(f'<line x1="{left}" y1="{y}" x2="{left+plotw}" y2="{y}" stroke="#d5dce1"/>');out.append(txt(left,y-4,f'{v:g}',8,'#5c6770'))
  if baseline is not None:
   y=bottom-ph*baseline/maxv;out.append(f'<line x1="{left}" y1="{y}" x2="{left+plotw}" y2="{y}" stroke="#a65936" stroke-dasharray="4,3"/>')
  bw=plotw/len(values)*.58
  for i,(v,nm) in enumerate(zip(values,names)):
   x=left+(i+.5)*plotw/len(values);h=ph*v/maxv
   out.append(f'<rect x="{x-bw/2}" y="{bottom-h}" width="{bw}" height="{h}" fill="{COL[i]}"/>');out.append(f'<rect x="{x-22}" y="{bottom-h-19}" width="44" height="13" fill="white"/>');out.append(txt(x,bottom-h-8,f'{v:.4f}' if lab!='Idiff' else f'{v:.2f}',10,anchor='middle'))
   for k,line in enumerate(nm.split('|')):out.append(txt(x,bottom+17+k*13,line,9,anchor='middle'))
 for i,s in enumerate(foot):out.append(txt(15,325+i*15,s,10))
 out.append('</svg>');(FIG/name).write_text('\n'.join(out))
chart('fig1_map_comparison.svg','The interaction extends to effect estimates','42 people; four contrasts; test-to-retest; same group atlas and estimator',[(m,[z['R_ORIG'][k],e['R_ORIG'][k]],['Historical|z-statistic','Effect|estimate'],mx,b) for m,k,mx,b in [('Rank-1','rank1',.8,1/42),('Idiff','idiff',60,None),('AUC','auc',1,.5)]],['Rank-1 and Idiff: p = 1/10001 in each representation. AUC is descriptive.','Dashed lines: rank-1 chance 1/42; AUC chance 0.5. No causal precision fraction inferred.'])
chart('fig2_rest_control.svg','Rest control: attenuation with residual evidence','Effect estimates; REST1 k = 10; endpoints are shown separately',[
('WM contrast rank-1',[r['baseline']['transfer']['rank1'],r['primary_REST1']['k10']['transfer']['rank1']],['Before','After'],.8,1/42),
('Interaction rank-1',[e['R_ORIG']['rank1'],a['R_REST']['before']['rank1']],['Before','After'],.8,1/42),
('Specificity AUC(SS,DS)',[r['baseline']['specificity']['auc_ss_ds'],r['primary_REST1']['k10']['specificity']['pooled']['auc_ss_ds']],['Before','After'],1,.5)],['After-control rank-1 p: WM contrast 1/2001; pooled interaction 1/10001.','AUC increase reflects changing similarity distributions; it is not increased retrieval.'])
def _att(k):return 100*(1-a[k]['after']['idiff']/a[k]['before']['idiff'])
def _pfrac(p,n):
 c=round(p*(n+1));return f'{c}/{n+1}'
_pa=[_pfrac(a[k]['attribution']['p_attrib_idiff'],a[k]['attribution']['b_shuf']) for k in ('R_ORIG','R_REST')]
FIG3_NOTE=f"Idiff attenuation: {_att('R_ORIG'):.1f}% (original), {_att('R_REST'):.1f}% (rest-controlled); attribution p = "+(f"{_pa[0]} each." if _pa[0]==_pa[1] else f"{_pa[0]} and {_pa[1]}.")
assert all(a[k][s]['p_rank1']==a[k][s]['p_idiff']==1/10001 for k in ('R_ORIG','R_REST') for s in ('after',)), 'Figure 3 footnote states p = 1/10001'
chart('fig3_anatomy_control.svg','Measured morphometry: partial attenuation','Effect-estimate interaction; LOSO anatomy prediction; k = 20, lambda = 1',[(m,[a['R_ORIG']['before'][k],a['R_ORIG']['after'][k],a['R_REST']['before'][k],a['R_REST']['after'][k]],['Original|before','Original|after','Rest|before','Rest|after'],mx,b) for m,k,mx,b in [('Rank-1','rank1',.8,1/42),('Idiff','idiff',60,None),('AUC','auc',1,.5)]],[FIG3_NOTE,'After-control rank-1 and Idiff p = 1/10001. Attenuation is not causal variance explained.'])
print('Wrote 3 figures to', FIG)
