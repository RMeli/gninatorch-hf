import os


def load_html(html_file: str) -> str:
    """
    Load file from HTML directory.

    Parameters
    ----------
    html_file: str
        HTML file name

    Returns
    -------
    str
        HTML file content
    """
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


def load_ligand_from_file(ligand_file) -> str:
    """
    Load ligand from file.

    Parameters
    ----------
    ligand_file: _TemporaryFileWrapper
        GradIO file object

    Returns
    -------
    str
        Ligand SDF file content
    """
    with open(ligand_file.name, "r") as f:
        return f.read()


def protein_html_from_file(protein_file) -> str:
    """
    Wrap 3Dmol.js code around protein PDB file.

    Parameters
    ----------
    protein_file: _TemporaryFileWrapper
        GradIO file object

    Returns
    -------
    str
        3Dmol.js HTML code for displaying a PDB file
    """
    protein = load_protein_from_file(protein_file)
    protein_html = load_html("protein.html")

    html = protein_html.replace("%%%PDB%%%", protein)

    wrapper = load_html("wrapper.html")

    return wrapper.replace("%%%HTML%%%", html)


def ligand_html_from_file(ligand_file) -> str:
    """
    Wrap 3Dmol.js code around ligand SDF file.

    Parameters
    ----------
    ligand_file: _TemporaryFileWrapper
        GradIO file object

    Returns
    -------
    str
        3Dmol.js HTML code for displaying a SDF file
    """

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


def predict(protein_file, ligand_file, cnn: str = "default"):
    """
    Run gnina-torch on protein-ligand complex.

    Parameters
    ----------
    protein_file: _TemporaryFileWrapper
        GradIO file object
    ligand_file: _TemporaryFileWrapper
        GradIO file object
    cnn: str
        CNN model to use

    Returns
    -------
    dict[str, float]
        CNNscore, CNNaffinity, and CNNvariance
    """
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

    # FIXME: Do this properly... =( [Might require light gnina-torch refactoring]
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
    ).round(6)


if __name__ == "__main__":
    import gradio as gr

    demo = gr.Blocks()

    with demo:
        gr.Markdown("# Gnina-Torch")
        gr.Markdown(
            "Score your protein-ligand compex and predict the binding affinity with [Gnina]"
            + "(https://github.com/gnina/gnina)'s scoring function. Poewerd by [gnina-torch]"
            + "(https://github.com/RMeli/gnina-torch), a PyTorch implementation of Gnina's"
            + " scoring function."
        )

        gr.Markdown("## Protein and Ligand")
        gr.Markdown(
            "Upload your protein and ligand files in PDB and SDF format, respectively."
        )
        with gr.Row():
            with gr.Box():
                pfile = gr.File(file_count="single", label="Protein file (PDB)")
                pbtn = gr.Button("View")

                protein = gr.HTML()
                pbtn.click(fn=protein_html_from_file, inputs=[pfile], outputs=protein)

            with gr.Box():
                lfile = gr.File(file_count="single", label="Ligand file (SDF)")
                lbtn = gr.Button("View")

                ligand = gr.HTML()
                lbtn.click(fn=ligand_html_from_file, inputs=[lfile], outputs=ligand)

        gr.Markdown("## Protein-Ligand Complex")
        with gr.Row():
            plcomplex = gr.HTML()

            # TODO: Automatically display complex when both files are uploaded
            plbtn = gr.Button("View")
            plbtn.click(
                fn=protein_ligand_html_from_file,
                inputs=[pfile, lfile],
                outputs=plcomplex,
            )

        gr.Markdown("## Gnina-Torch")
        with gr.Row():
            dd = gr.Dropdown(
                choices=[
                    "default",
                    "redock_default2018_ensemble",
                    "general_default2018_ensemble",
                    "crossdock_default2018_ensemble",
                ],
                value="default",
                label="CNN model",
            )

            df = gr.Dataframe()
            btn = gr.Button("Score!")
            btn.click(fn=predict, inputs=[pfile, lfile, dd], outputs=df)

    demo.launch()
