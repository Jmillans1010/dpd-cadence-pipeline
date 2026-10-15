"""Evaluate the five saved postdistorters y/G -> x, without training or inversion."""
from pathlib import Path
import json
import shutil
import hashlib
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
RESULTS = BASE/'framework_dpd/data/dpd_results'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, help='Carpeta de ejecución o su nombre; por defecto latest_run.json.')
args = parser.parse_args()
if args.run is None:
    latest = json.loads((RESULTS/'latest_run.json').read_text(encoding='utf-8-sig'))
    RUN = Path(latest['directory'])
else:
    candidates = [args.run, BASE/args.run, RESULTS/args.run]
    RUN = next((p.resolve() for p in candidates if p.is_dir()), args.run.resolve())
MODELS = ['MP','EMP','GMP','PrunedVolterra','LUT']
FOLDERS = dict(MP='comparacion_mp', EMP='comparacion_emp', GMP='comparacion_gmp', PrunedVolterra='comparacion_volterra_pruned', LUT='comparacion_lut')
for folder in ['input','modelos','resultados']:
    (ROOT/folder).mkdir(parents=True,exist_ok=True)

def decode(o):
    if '__complex__' in o:
        return complex(*o['__complex__'])
    if '__ndarray__' in o:
        return np.array(o['values'],dtype=o['dtype'])
    if '__nonfinite__' in o:
        return float(o['__nonfinite__'])
    return o

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'),object_hook=decode)

def predict(model,z,idx):
    if model['model_name']=='LUT':
        nodes,values=model['lut_nodes'],model['lut_values']
        r=abs(z[idx])
        if model['config']['interpolation']=='nearest':
            cells=np.searchsorted((nodes[:-1]+nodes[1:])/2,r)
            return z[idx]*values[cells]
        if model['config']['interpolation']!='linear':
            raise ValueError('Interpolación LUT desconocida.')
        return z[idx]*(np.interp(r,nodes,values.real)+1j*np.interp(r,nodes,values.imag))
    result=np.zeros(len(idx),complex)
    for c,t in zip(model['coefficients'],model['basis_terms'],strict=True):
        if t['kind']=='envelope':
            v=z[idx-t['signal_delay']]*abs(z[idx-t['envelope_delay']])**t['envelope_power']
        else:
            assert t['kind']=='volterra'
            v=np.ones(len(idx),complex)
            for d in t['direct_delays']:
                v*=z[idx-d]
            for d in t['conjugate_delays']:
                v*=z[idx-d].conjugate()
        result+=c*v
    return result

def metrics(real,pred):
    mse=float(np.mean(abs(real-pred)**2))
    ratio=mse/float(np.mean(abs(real)**2))
    return dict(samples=len(real),nmse_db=float(10*np.log10(ratio)),rmse=float(np.sqrt(mse)),relative_rms_percent=float(100*np.sqrt(ratio)))

manifest=read(RUN/'manifest.json')
if manifest.get('status') not in ('completed','completed_with_unavailable_models'):
    raise ValueError(f'La ejecución no está terminada: {RUN}')
if manifest['provenance'].get('split_mode')!='training_folder_plus_chronological_validation_holdout':
    raise ValueError('Esta comparación requiere validation_holdout; no admite otras particiones.')
validation_source=next(s for s in manifest['provenance']['sources'] if s['role']=='validation')
sources=[Path(validation_source['path']), BASE/'framework_dpd/data/input/Validation/aligned_waveform.csv',
         ROOT/'input/aligned_waveform.csv']
source=next((p for p in sources if p.is_file() and
             hashlib.sha256(p.read_bytes()).hexdigest()==validation_source['sha256']),None)
if source is None:
    raise ValueError(f'No hay CSV que coincida con el hash de validation de {RUN}.')
saved_models={name:read(RUN/f'{name}.json') for name in MODELS}
for name,saved in saved_models.items():
    if saved.get('model') is None or saved['model']['model_name']!=name:
        raise ValueError(f'La ejecución no contiene un modelo válido para {name}: {RUN}')
# Refresh the complete snapshot on every run: never mix a new manifest with
# existing coefficients from another experiment. Validate sources before copying.
for source_path,dest in [(RUN/'manifest.json',ROOT/'input/manifest.json'),
                         (source,ROOT/'input/aligned_waveform.csv')]+[
                         (RUN/f'{name}.json',ROOT/f'modelos/{name}.json') for name in MODELS]:
    if source_path.resolve()!=dest.resolve():
        shutil.copy2(source_path,dest)
print(f'Comparando modelos de {RUN.name}',flush=True)
gain=manifest['gain']
table=np.genfromtxt(ROOT/'input/aligned_waveform.csv',delimiter=',',names=True)
x=table['I_in']+1j*table['Q_in']
y=table['I_out']+1j*table['Q_out']
z=y/gain
p=manifest['provenance']
guard,past,future=p['guard_samples_each_edge'],p['common_past'],p['common_future']
boundary=p['validation_boundary']
indices={'validation':np.arange(guard+past,boundary-guard-future),
         'test':np.arange(boundary+2*guard+past,len(x)-guard-future)}
for partition,idx in indices.items():
    expected_samples=p['samples'][partition]['scored']
    if len(idx)!=expected_samples:
        raise ValueError(f'{partition}: índices incompatibles con el manifiesto ({len(idx)} frente a {expected_samples}).')
summary={}
predictions={}
for name in MODELS:
    saved=saved_models[name]
    model=saved['model']
    result={}
    for partition,idx in indices.items():
        estimated=predict(model,z,idx)
        result[partition]=metrics(x[idx],estimated)
        # Reproduce the scores saved when these exact models were trained.
        expected=saved[f'nmse_{partition}_db']
        actual=result[partition]['nmse_db']
        if not np.isclose(actual,expected,rtol=0,atol=1e-8):
            raise ValueError(f'{name}/{partition}: NMSE recalculado {actual} != guardado {expected}; ejecución {RUN.name}.')
        result[partition]['original_nmse_db']=expected
        columns=np.column_stack([idx,table['time_reference_s'][idx],y[idx].real,y[idx].imag,z[idx].real,z[idx].imag,x[idx].real,x[idx].imag,estimated.real,estimated.imag,(x[idx]-estimated).real,(x[idx]-estimated).imag])
        target=ROOT/f'resultados/{name}'
        target.mkdir(exist_ok=True)
        np.savetxt(target/f'comparacion_{partition}.csv',columns,delimiter=',',header='sample,time_s,I_out_measured,Q_out_measured,I_out_normalized,Q_out_normalized,I_input_real,Q_input_real,I_input_reconstructed,Q_input_reconstructed,error_I,error_Q',comments='')
        if partition=='test':
            predictions[name]=estimated
    result['direction']='measured y / original G -> reconstructed x'
    result['config']=saved['best_config']
    result['num_coefficients']=len(model['coefficients'])
    (target/'metricas.json').write_text(json.dumps(result,indent=2))
    summary[name]=result

idx=indices['test']
real=x[idx]
reference_limit=1.08*np.max(np.abs(np.r_[real.real,real.imag]))
model_limits={name:1.08*np.max(np.abs(np.r_[v.real,v.imag])) for name,v in predictions.items()}
# Keep divergent predictions visible without flattening the other panels.
limit=max([reference_limit,*[v for v in model_limits.values() if v<=3*reference_limit]])
expanded={name for name,v in model_limits.items() if v>3*reference_limit}
def setup(ax,title,axis_limit=None):
    axis_limit=limit if axis_limit is None else axis_limit
    ax.set(title=title,xlabel='I (unidades del CSV)',ylabel='Q (unidades del CSV)',xlim=(-axis_limit,axis_limit),ylim=(-axis_limit,axis_limit))
    ax.set_aspect('equal',adjustable='box')
    ax.grid(alpha=.2)
def scatter(ax,v,color,label):
    ax.scatter(v.real,v.imag,s=7,alpha=.35,color=color,label=label,rasterized=True)

fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
setup(axes.flat[0],'Entrada real (referencia)')
scatter(axes.flat[0],real,'#0072B2','Entrada real')
for ax,name in zip(list(axes.flat)[1:],MODELS):
    m=summary[name]['test']
    scale_note='\nEscala ampliada por predicciones extremas' if name in expanded else ''
    setup(ax,f"{name} · NMSE {m['nmse_db']:.2f} dB · RMS {m['relative_rms_percent']:.2f}%{scale_note}",
          model_limits[name] if name in expanded else limit)
    scatter(ax,real,'#0072B2','Entrada real')
    scatter(ax,predictions[name],'#D55E00','Entrada reconstruida')
    ax.legend(fontsize=8)
fig.suptitle(f'Cinco modelos: salida medida / G → entrada reconstruida\nTest: {len(idx)} muestras · {RUN.name} · sin reentrenamiento',fontsize=14)
fig.savefig(ROOT/'resultados/constelaciones_cinco_modelos.png',dpi=160)
fig.savefig(ROOT/'resultados/constelaciones_cinco_modelos.svg')
plt.close(fig)

for name in MODELS:
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,title in zip(axes,['Entrada real',f'Entrada reconstruida: {name}','Superposición']):
        setup(ax,title,model_limits[name] if name in expanded else limit)
    for ax in [axes[0],axes[2]]:
        scatter(ax,real,'#0072B2','Real')
    for ax in [axes[1],axes[2]]:
        scatter(ax,predictions[name],'#D55E00','Reconstruida')
    axes[2].legend()
    fig.suptitle(f"{name}: y/G → x · NMSE {summary[name]['test']['nmse_db']:.2f} dB\nTodas las muestras del test; incluye transiciones entre símbolos")
    for ext in ['png','svg']:
        fig.savefig(ROOT/f'resultados/{name}/constelacion.{ext}',dpi=180)
    plt.close(fig)

summary['_provenance']={'run':RUN.name,'gain':[gain.real,gain.imag],
                        'input_sha256':hashlib.sha256((ROOT/'input/aligned_waveform.csv').read_bytes()).hexdigest(),
                        'test_first_index':int(idx[0]),'test_last_index':int(idx[-1]),
                        'verification':'All ten validation/test NMSE values match original saved results to 1e-8 dB'}
(ROOT/'resultados/metricas_cinco_modelos.json').write_text(json.dumps(summary,indent=2))
report=f'''# Comparación: salida medida a entrada reconstruida

Se evalúan directamente los cinco modelos de {RUN.name},
sin inversión numérica ni reentrenamiento. Entrada del modelo: z=y/G, usando la ganancia
compleja del manifiesto original. Referencia: entrada medida x. Predicción: x_estimada=F(z).
Esto evalúa el modelo inverso como postdistorsionador; no demuestra por sí mismo
la linealización obtenida al aplicar un predistorsionador antes del amplificador.

Se conserva la alineación original. Todos usan los mismos {len(indices['test'])} índices de test y
{len(indices['validation'])} índices de validación, guardas y contexto del manifiesto (incluye adelantos GMP).
Los diez NMSE reproducen los valores guardados durante el entrenamiento con tolerancia 1e-8 dB.
Criterio de selección: {manifest['config']['optimization'].get('stopping_rule','reference_fraction')}.
En BIC, validación es diagnóstica; en reference_fraction participa en selección.
Test es la segunda mitad reservada del registro Validation.

| Modelo | NMSE test dB | Error RMS relativo % | RMSE | NMSE validación dB |
|---|---:|---:|---:|---:|
'''
for name in MODELS:
    m=summary[name]['test']
    report+=f"| {name} | {m['nmse_db']:.6f} | {m['relative_rms_percent']:.6f} | {m['rmse']:.6g} | {summary[name]['validation']['nmse_db']:.6f} |\n"
report+='''
NMSE=10 log10(sum(|x-x_estimada|²)/sum(|x|²)). Error RMS relativo=100 sqrt(error
normalizado); no es EVM contra símbolos ideales. RMSE en unidades del CSV.
Los gráficos usan escala común salvo paneles con predicciones extremas, que se
muestran completos con escala ampliada indicada en el título. Usan todas las muestras temporales,
incluidas transiciones; no hay nuevo ajuste de ganancia, fase, retardo o instante de símbolo.

Los resultados anteriores de inversión de MP, LUT y Volterra se preservan en
historico_inversion/ dentro de cada workspace antiguo. Sus comparar.py, README y
resultados actuales ahora corresponden a este ejercicio corregido.

Ejecutar: python comparar.py (numpy y matplotlib). Por defecto usa latest_run.json.
Para otra ejecución: python comparar.py --run run_<fecha>. Las copias en input/ y modelos/
se actualizan juntas desde esa ejecución; el CSV se verifica por SHA-256.
Cada modelo tiene CSV separados de validación y test, métricas y constelación en resultados/.
'''
(ROOT/'README.md').write_text(report,encoding='utf-8')
(ROOT/'comparacion_cinco_modelos.code-workspace').write_text(json.dumps({'folders':[{'path':'.'}]},indent=2))
# Preserve previous inverse experiments before updating their current entry points.
for name,folder in FOLDERS.items():
    dest=BASE/folder
    dest.mkdir(exist_ok=True)
    archive=dest/'historico_inversion'
    if name in ['MP','LUT','PrunedVolterra'] and not archive.exists():
        archive.mkdir()
        for item in ['comparar.py','README.md','resultados']:
            src=dest/item
            if src.is_dir():
                shutil.copytree(src,archive/item)
            elif src.exists():
                shutil.copy2(src,archive/item)
    shutil.copytree(ROOT/f'resultados/{name}',dest/'resultados',dirs_exist_ok=True)
    # Replace the old generic CSV and metric names with the corrected test results.
    shutil.copy2(ROOT/f'resultados/{name}/comparacion_test.csv',dest/'resultados/comparacion.csv')
    (dest/'README.md').write_text(f'# {name}: comparación corregida\n\nAhora se evalúa y/G → x, sin invertir ni reentrenar.\n\n'+report,encoding='utf-8')
    (dest/'comparar.py').write_text("from pathlib import Path\nimport runpy\nrunpy.run_path(str(Path(__file__).resolve().parent.parent/'comparacion_cinco_modelos/comparar.py'),run_name='__main__')\n",encoding='utf-8')
    (dest/f'{folder}.code-workspace').write_text(json.dumps({'folders':[{'path':'.'}]},indent=2))
print(json.dumps({name:summary[name]['test'] for name in MODELS},indent=2))
