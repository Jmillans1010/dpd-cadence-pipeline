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
RUN = ROOT/'modelos'
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
sources=[ROOT/'input/aligned_waveform.csv']
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


import plotly.graph_objects as go
from plotly.subplots import make_subplots
idx=indices['test']
real=x[idx]
origin=json.loads((ROOT/'procedencia.json').read_text())['source_run']
ref_limit=1.15*np.max(np.abs(np.r_[real.real,real.imag]))
full_limit=1.05*max(np.max(np.abs(np.r_[v.real,v.imag])) for v in [real,*predictions.values()])
names=['Entrada real',*MODELS]
titles=['Entrada real (referencia)']+[f"{n} · NMSE {summary[n]['test']['nmse_db']:.2f} dB" for n in MODELS]
fig=make_subplots(rows=2,cols=3,subplot_titles=titles,horizontal_spacing=.065,vertical_spacing=.16)
for k,name in enumerate(names):
    row,col=divmod(k,3)
    fig.add_trace(go.Scattergl(x=real.real,y=real.imag,mode='markers',name='Entrada real',legendgroup='real',showlegend=k==0,marker=dict(size=4,color='#0072B2',opacity=.5),hovertemplate='I=%{x:.6g} V<br>Q=%{y:.6g} V<extra>Entrada real</extra>'),row=row+1,col=col+1)
    if k:
        v=predictions[name]
        fig.add_trace(go.Scattergl(x=v.real,y=v.imag,mode='markers',name='Entrada reconstruida',legendgroup='reconstruida',showlegend=k==1,marker=dict(size=4,color='#D55E00',opacity=.5),hovertemplate='I=%{x:.6g} V<br>Q=%{y:.6g} V<extra>'+name+'</extra>'),row=row+1,col=col+1)
    fig.update_xaxes(title_text='I (V)',range=[-full_limit,full_limit],row=row+1,col=col+1)
    fig.update_yaxes(title_text='Q (V)',range=[-full_limit,full_limit],scaleanchor='x'+(str(k+1) if k else ''),scaleratio=1,row=row+1,col=col+1)
def ranges(limit):
    return {axis+(str(k) if k>1 else '')+'.range':[-limit,limit] for axis in ['xaxis','yaxis'] for k in range(1,7)}
fig.update_layout(height=920,title='Comparador de cinco modelos · y/G → entrada reconstruida',template='plotly_white',legend=dict(orientation='h',y=1.08),margin=dict(t=150),updatemenus=[dict(type='buttons',direction='right',x=0,y=1.14,buttons=[dict(label='Escala común: todos los puntos',method='relayout',args=[ranges(full_limit)]),dict(label='Zoom entrada (puede ocultar extremos)',method='relayout',args=[ranges(ref_limit)])])])
rows=''.join(f"<tr><td>{n}</td><td>{summary[n]['test']['nmse_db']:.3f}</td><td>{summary[n]['test']['relative_rms_percent']:.3f}</td><td>{summary[n]['test']['rmse']:.6g}</td></tr>" for n in MODELS)
html='''<!doctype html><html lang="es"><meta charset="utf-8"><title>Comparador de constelaciones</title><style>body{font:16px system-ui;margin:30px;color:#172033}table{border-collapse:collapse}td,th{padding:10px 24px;border-bottom:1px solid #ddd;text-align:right}h1{font-size:26px}</style><h1>Entrada real frente a entrada reconstruida</h1>'''
html+=f'<p>Salida medida → división por G original → modelo → entrada reconstruida. Sin inversión ni reentrenamiento. Unidades I/Q: <b>voltios</b>.</p><p>Test: {len(idx)} muestras · Ejecución: {origin}. Se incluyen transiciones entre símbolos. Azul: entrada real; naranja: reconstrucción.</p><p>La vista inicial usa la misma escala y muestra todos los puntos. El botón de zoom permite observar la zona de entrada; los extremos permanecen en los datos. Arrastra para ampliar y haz doble clic para restablecer.</p>'
html+=fig.to_html(full_html=False,include_plotlyjs=True,config={'responsive':True,'displaylogo':False})
html+='<h2>Métricas del test</h2><table><tr><th>Modelo</th><th>NMSE (dB)</th><th>RMS relativo (%)</th><th>RMSE (V)</th></tr>'+rows+'</table><p>RMS relativo respecto a la entrada medida; no es EVM contra símbolos ideales. Esta evaluación de postdistorsión no mide por sí sola la linealización del amplificador.</p></html>'
(ROOT/'resultados/comparador.html').write_text(html,encoding='utf-8')
for view,limit in [('completa',full_limit),('zoom_entrada',ref_limit)]:
    fig2,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
    for k,(ax,name) in enumerate(zip(axes.flat,names)):
        ax.scatter(real.real,real.imag,s=6,alpha=.4,color='#0072B2',label='Entrada real')
        clipped=0
        if k:
            v=predictions[name]
            ax.scatter(v.real,v.imag,s=6,alpha=.4,color='#D55E00',label='Reconstruida')
            clipped=int(np.sum((abs(v.real)>limit)|(abs(v.imag)>limit)))
        title=titles[k]+(f'\n{clipped} puntos fuera de vista' if clipped else '')
        ax.set(title=title,xlabel='I (V)',ylabel='Q (V)',xlim=(-limit,limit),ylim=(-limit,limit))
        ax.set_aspect('equal',adjustable='box'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig2.suptitle(f'Entrada real y reconstrucciones · {len(idx)} muestras de test\n'+('Escala común completa' if view=='completa' else 'Zoom común alrededor de la entrada; extremos indicados'),fontsize=15)
    fig2.savefig(ROOT/f'resultados/constelaciones_{view}.png',dpi=170)
    fig2.savefig(ROOT/f'resultados/constelaciones_{view}.svg')
    plt.close(fig2)
summary['_provenance']={'source_run':origin,'gain':[gain.real,gain.imag],'units':'V','input_sha256':validation_source['sha256'],'verification':'10 NMSE validation/test reproduced within 1e-8 dB'}
(ROOT/'resultados/metricas.json').write_text(json.dumps(summary,indent=2))
report=f'''# Comparador independiente de cinco modelos

Abrir resultados/comparador.html: panel interactivo autónomo, funciona sin internet.
Ejecutar ejecutar.ps1 o `python comparar.py` para regenerarlo.
Dependencias: `python -m pip install -r requirements.txt`.

Se incluyen copias de los cinco modelos, el manifiesto y el CSV de {origin}.
No se consulta ni modifica ningún otro workspace al ejecutar el comparador.
Entrada del modelo: salida medida y dividida por la ganancia compleja G original.
Predicción: entrada reconstruida x_estimada. Referencia: entrada medida x.
I y Q se expresan en voltios; RMSE también en voltios. No se normalizan las
constelaciones gráficas a potencia unitaria. No hay nuevos ajustes de fase o ganancia.

Test: {len(idx)} muestras. Se verifican los diez NMSE de validación/test frente
a los resultados originales, con tolerancia 1e-8 dB. La validación participó en
selección de modelos; test corresponde al segmento reservado del mismo registro.
Se incluyen todas las muestras temporales y transiciones, no solo centros de símbolo.

La vista completa mantiene la misma escala para todos y muestra todos los puntos.
El zoom alrededor de la entrada puede ocultar extremos, identificados en la figura PNG.
Los resultados de Pruned Volterra en esta ejecución tienen errores muy grandes;
se preservan íntegramente. La escala completa permite examinarlos.

Archivos: resultados/comparador.html, constelaciones_completa.png/svg,
constelaciones_zoom_entrada.png/svg, metricas.json y CSV por modelo/partición.
NMSE=10log10(sum(|x-x_estimada|²)/sum(|x|²)). RMS relativo=100sqrt(error normalizado).
No es EVM contra símbolos ideales ni prueba de linealización con DPD antes del PA.
'''
(ROOT/'README.md').write_text(report,encoding='utf-8')
print(json.dumps({n:summary[n]['test'] for n in MODELS},indent=2))
