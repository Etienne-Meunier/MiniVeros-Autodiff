import os
import subprocess
import webbrowser
import treescope

class OceanModelAutovisualizer(treescope.Autovisualizer):
    def __call__(self, value, path, *args, **kwargs):
        from treescope import type_registries
        adapter = type_registries.lookup_ndarray_adapter(value)
        
        if adapter is not None:
            shape = getattr(value, "shape", None)
            if shape is not None:
                ndim = len(shape)
                
                if ndim >= 2:
                    # Écrit sans crochets directs pour contourner mon bug de génération
                    render_kwargs = dict(columns=list((0,)), rows=list((1,)))
                    if ndim > 2:
                        render_kwargs["sliders"] = list(range(2, ndim))
                        
                    figure = treescope.render_array(value, **render_kwargs)
                    return treescope.IPythonVisualization(figure)
        return None

def browse(obj, filename="treescope_render.html"):
    """Génère l'HTML et force Arc Browser à recharger le même onglet s'il existe."""
    with treescope.active_autovisualizer.set_scoped(OceanModelAutovisualizer()):
        html_content = treescope.render_to_html(obj)
        
    file_path = os.path.abspath(filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Script d'automatisation pour Arc Browser (macOS)
    applescript = f'''
    tell application "Arc"
        set found to false
        tell front window
            repeat with t in tabs
                if URL of t contains "{filename}" then
                    tell t to reload
                    set found to true
                    exit repeat
                end if
            end repeat
            if not found then
                make new tab with properties {{URL:"file://{file_path}"}}
            end if
        end tell
    end tell
    '''
    
    try:
        subprocess.run(["osascript", "-e", applescript], check=True, capture_output=True)
    except Exception:
        webbrowser.open(f"file://{file_path}", new=0)
