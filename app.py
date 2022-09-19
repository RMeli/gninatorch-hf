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

    wrapper = load_html("wrapper.html")

    return wrapper.replace("%%%HTML%%%", html)


def ligand_html_from_file(ligand_file):
    ligand = load_ligand_from_file(ligand_file)
    ligand_html = load_html("ligand.html")

    html = ligand_html.replace("%%%SDF%%%", ligand)

    wrapper = load_html("wrapper.html")

    return wrapper.replace("%%%HTML%%%", html)


def protein_ligand_html_from_file(protein_file, ligand_file):
    protein = load_protein_from_file(protein_file)
    ligand = load_ligand_from_file(ligand_file)
    protein_ligand_html = load_html("pl.html")

    html = protein_ligand_html.replace("%%%PDB%%%", protein)
    html = html.replace("%%%SDF%%%", ligand)

    wrapper = load_html("wrapper.html")

    return wrapper.replace("%%%HTML%%%", html)


def predict(protein_file, ligand_file, cnn="default"):
    import molgrid
    from gninatorch import gnina, dataloaders
    import torch
    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model, ensemble = gnina.setup_gnina_model(cnn, 23.5, 0.5)
    model.eval()
    model.to(device)

    example_provider = molgrid.ExampleProvider(
        data_root="",
        balanced=False,
        shuffle=False,
        default_batch_size=1,
        iteration_scheme=molgrid.IterationScheme.SmallEpoch,
    )

    with open("data.in", "w") as f:
        f.write(protein_file.name)
        f.write(" ")
        f.write(ligand_file.name)

    print("Populating example provider... ", end="")
    example_provider.populate("data.in")
    print("done")

    grid_maker = molgrid.GridMaker(resolution=0.5, dimension=23.5)

    # TODO: Allow average over different rotations
    loader = dataloaders.GriddedExamplesLoader(
        example_provider=example_provider,
        grid_maker=grid_maker,
        random_translation=0.0,  # No random translations for inference
        random_rotation=False,  # No random rotations for inference
        grids_only=True,
        device=device,
    )

    print("Loading and gridding data... ", end="")
    batch = next(loader)
    print("done")

    print("Predicting... ", end="")
    with torch.no_grad():
        log_pose, affinity, affinity_var = model(batch)
    print("done")

    return pd.DataFrame(
        {
            "CNNscore": [torch.exp(log_pose[:, -1]).item()],
            "CNNaffinity": [affinity.item()],
            "CNNvariance": [affinity_var.item()],
        }
    )


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

    gr.Markdown("# Protein-Ligand Complex")
    with gr.Row():
        plcomplex = gr.HTML()

        # TODO: Automatically display complex when both files are uploaded
        plbtn = gr.Button("View")
        plbtn.click(
            fn=protein_ligand_html_from_file, inputs=[pfile, lfile], outputs=plcomplex
        )

    gr.Markdown("# Gnina-Torch")
    with gr.Row():
        df = gr.Dataframe()
        btn = gr.Button("Score!")
        btn.click(fn=predict, inputs=[pfile, lfile], outputs=df)


demo.launch()
