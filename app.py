import gradio as gr

import os

def load_html(html_file: str):
    with open(os.path.join("html", html_file), "r") as f:
        return f.read()

def load_protein_from_file(protein_file) -> str:
    """
    Parameters
    ----------
    protein_file: _TemporaryFileWrapper
        GradIO file object

    Returns
    -------
    str
        Protein PDB file content
    """
    with open(protein_file.name, "r") as f:
        return f.read()

def load_ligand_from_file(ligand_file):
    with open(ligand_file.name, "r") as f:
        return f.read()
    
def protein_html_from_file(protein_file):
    protein = load_protein_from_file(protein_file)
    protein_html = load_html("protein.html")

    html = protein_html.replace("%%%PDB%%%", protein)

    return f"""<iframe style="width: 100%; height: 600px" name="result" allow="midi; geolocation; microphone; camera; 
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
    allow-scripts allow-same-origin allow-popups 
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
    allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""

def ligand_html_from_file(ligand_file):
    ligand = load_ligand_from_file(ligand_file)
    ligand_html = load_html("ligand.html")

    html = ligand_html.replace("%%%SDF%%%", ligand)

    return f"""<iframe style="width: 100%; height: 600px" name="result" allow="midi; geolocation; microphone; camera; 
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
    allow-scripts allow-same-origin allow-popups 
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
    allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""

def protein_ligand_html_from_file(protein_file, ligand_file):
    protein = load_protein_from_file(protein_file)
    ligand = load_ligand_from_file(ligand_file)
    protein_ligand_html = load_html("pl.html")

    html = protein_ligand_html.replace("%%%PDB%%%", protein)
    html = html.replace("%%%SDF%%%", ligand)

    return f"""<iframe style="width: 100%; height: 600px" name="result" allow="midi; geolocation; microphone; camera; 
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
    allow-scripts allow-same-origin allow-popups 
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
    allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""

demo = gr.Blocks()

with demo:
    gr.Markdown("# Protein and Ligand")
    with gr.Row():
        with gr.Box():
            pfile = gr.File(file_count="single")
            pbtn = gr.Button("View")

            protein = gr.HTML()
            pbtn.click(fn=protein_html_from_file, inputs=[pfile], outputs=protein)

        with gr.Box():
            lfile = gr.File(file_count="single")
            lbtn = gr.Button("View")

            ligand = gr.HTML()
            lbtn.click(fn=ligand_html_from_file, inputs=[lfile], outputs=ligand)

    with gr.Row():
        gr.Markdown("# Protein-Ligand Complex")
        plcomplex = gr.HTML()

        # TODO: Automatically display complex when both files are uploaded
        plbtn = gr.Button("View")
        plbtn.click(fn=protein_ligand_html_from_file, inputs=[pfile, lfile], outputs=plcomplex)

demo.launch()
