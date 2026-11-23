"""Validate every task, then summarize fully paired seed-level sensitivity."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.stats import bootstrap
import study

def interval(values):
    values=np.asarray(values,dtype=float)
    if len(values)!=20 or not np.isfinite(values).all(): raise ValueError('Expected 20 finite seed values')
    if np.ptp(values)<1e-14: return float(values.mean()),float(values.mean()),'constant'
    result=bootstrap((values,),np.mean,n_resamples=10000,confidence_level=0.95,method='BCa',random_state=20260924)
    lo,hi=float(result.confidence_interval.low),float(result.confidence_interval.high)
    if not np.isfinite([lo,hi]).all(): raise ValueError('Nonfinite BCa interval; inspect rather than silently replace')
    return lo,hi,'BCa'

def csv_write(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(output):
    output=Path(output).resolve();study.configure();plan=study.read_plan(output,False)
    results={};endpoints=[];routes=[]
    for task in study.tasks():
        r=study.verify_result(output,task['index']);key=(task['setting'],task['seed'],task['scenario'])
        results[key]=r['results']
        for mode,ep in r['results'].items():
            endpoints.append({'setting':task['setting'],'seed':task['seed'],'scenario':task['scenario'],'mode':mode,
                              **{k:ep[k] for k in study.METRICS}})
    rows=[];settings=study.design()
    for setting in settings:
        sid=setting['id']
        for scenario in study.SCENARIOS+['overall']:
            scs=study.SCENARIOS if scenario=='overall' else [scenario]
            values={m:np.array([np.mean([results[(sid,seed,sc)][m]['ari'] for sc in scs]) for seed in study.SEEDS]) for m in study.MODES}
            for comparator in ['no_context','static']:
                gain=values['agribrain']-values[comparator]
                nominal=np.array([np.mean([results[('nominal',seed,sc)]['agribrain']['ari']-results[('nominal',seed,sc)][comparator]['ari'] for sc in scs]) for seed in study.SEEDS])
                lo,hi,method=interval(gain);change=gain-nominal;clo,chi,cmethod=interval(change)
                rows.append({'setting':sid,'family':setting['family'],'scenario':scenario,'comparator':comparator,
                    'static_ari':float(values['static'].mean()),'no_context_ari':float(values['no_context'].mean()),'agribrain_ari':float(values['agribrain'].mean()),
                    'mean_gain':float(gain.mean()),'ci_low':lo,'ci_high':hi,'ci_method':method,
                    'relative_gain_percent':100*float(gain.mean()/values[comparator].mean()),
                    'change_from_nominal':float(change.mean()),'change_ci_low':clo,'change_ci_high':chi,'change_ci_method':cmethod,
                    'negative_seed_gains':int(sum(gain<0)),'mean_ranking_reversal':bool(gain.mean()<0)})
        for seed in study.SEEDS:
            for sc in study.SCENARIOS:
                r=results[(sid,seed,sc)];nominal=results[('nominal',seed,sc)]
                routes.append({'setting':sid,'seed':seed,'scenario':sc,
                    'agribrain_vs_no_context_difference_pct':100*float(np.mean(np.asarray(r['agribrain']['action_trace'])!=np.asarray(r['no_context']['action_trace']))),
                    **{f'{m}_change_from_nominal_pct':100*float(np.mean(np.asarray(r[m]['action_trace'])!=np.asarray(nominal[m]['action_trace']))) for m in study.MODES}})
    out=output/'analysis';out.mkdir(exist_ok=True)
    csv_write(out/'seed_endpoints.csv',endpoints);csv_write(out/'paired_ari_sensitivity.csv',rows);csv_write(out/'route_changes.csv',routes)
    csv_write(out/'effective_parameters.csv',[{'setting':s['id'],**{k+'_factor':v for k,v in s['factors'].items()},**s['effective_policy']} for s in settings])
    target=[r for r in rows if r['comparator']=='no_context'];overall=[r for r in target if r['scenario']=='overall']
    summary={'counts':plan['counts'],'intervals':'pointwise 95% BCa; not simultaneous',
             'minimum_scenario_gain':min((r for r in target if r['scenario']!='overall'),key=lambda r:r['mean_gain']),
             'minimum_overall_gain':min(overall,key=lambda r:r['mean_gain']),
             'overall_settings_lower_ci_above_zero':sum(r['ci_low']>0 for r in overall),
             'overall_settings_lower_ci_above_001':sum(r['ci_low']>0.01 for r in overall),
             'mean_ranking_reversals':[r for r in target if r['mean_ranking_reversal']]}
    study.save(out/'summary.json',summary)
    plot(out,[r for r in overall if r['family']=='weights'])
    study.save(output/'COMPLETE.json',{'plan_sha256':study.sha(output/'plan.json'),'tasks_verified':2300,
        'analysis_files':{p.name:study.sha(p) for p in out.iterdir() if p.is_file()}})
    print('COMPLETE: 23 weight settings, 2300 tasks, 20700 study episodes. See analysis/summary.json.')

def plot(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':12,'axes.titlesize':13,'figure.titlesize':16,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(12,10),sharey=True)
    y=np.arange(len(rows))
    for ax,mean,low,high,color in [(axes[0],'mean_gain','ci_low','ci_high','#009E73'),(axes[1],'change_from_nominal','change_ci_low','change_ci_high','#0072B2')]:
        for yi,r in zip(y,rows):
            # Draw exact CI endpoints even for an asymmetric interval not containing the estimate.
            ax.hlines(yi,r[low],r[high],color=color,lw=1.5)
            ax.plot([r[low],r[high]],[yi,yi],marker='|',linestyle='',color=color)
            ax.plot(r[mean],yi,marker='o' if yi else 'D',color=color,ms=5)
        ax.axvline(0,color='0.3',ls='--',lw=1)
        ax.grid(axis='x',color='0.9');ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0].axvline(.01,color='0.6',ls=':',lw=1)
    axes[0].set_yticks(y,[r['setting'].replace('_',' ').replace('minus20','−20%').replace('plus20','+20%') for r in rows]);axes[0].invert_yaxis()
    axes[0].set_title('(a) AGRI-BRAIN advantage');axes[0].set_xlabel('Paired ARI gain over No-Context')
    axes[1].set_title('(b) Dependence on weight settings');axes[1].set_xlabel('Change in paired gain from nominal')
    fig.suptitle('Sensitivity of resilience gains to selected weights',y=.985)
    fig.tight_layout(rect=(0,0,1,.965))
    for ext in ['png','pdf','svg']: fig.savefig(out/f'Weight_Sensitivity.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();run(args.output)
