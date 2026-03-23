"""
Jupyter ipywidgets for PDE Latent Representation demonstrator.

Provides interactive widgets for PDE solution generation, autoencoder
configuration, and execution mode control in Jupyter notebooks.

Follows the UC2/UC5 widget pattern with helper functions,
tuple-based build_widgets, and individual-widget get_args_from_widgets.
"""

import ipywidgets as widgets
import argparse


def create_dropdown(options, value, description):
    """
    Helper function to create a Dropdown widget.
    """
    return widgets.Dropdown(options=options, value=value, description=description)


def create_select_multiple(options, value, description):
    """
    Helper function to create a SelectMultiple widget.
    """
    return widgets.SelectMultiple(options=options, value=value, description=description)


def create_int_slider(value, min_val, max_val, step, description):
    """
    Helper function to create an IntSlider widget.
    """
    return widgets.IntSlider(
        value=value, min=min_val, max=max_val, step=step, description=description,
    )


def create_float_log_slider(value, min_val, max_val, step, description, readout_format='.2e'):
    """
    Helper function to create a FloatLogSlider widget.
    """
    return widgets.FloatLogSlider(
        value=value, min=min_val, max=max_val, step=step,
        description=description, readout_format=readout_format,
    )


def create_int_input(value, description):
    """
    Helper function to create an IntText widget.
    """
    return widgets.IntText(value=value, description=description)


def build_widgets():
    """
    Build all widgets for PDE latent representation experiment configuration.

    Returns:
        Tuple of widgets: (n_sol, grid_levels, n_sf, vel_scale, latent_dim,
                           epochs, batch_size, seed)
    """
    n_sol_widget = create_int_slider(200, 10, 1000, 10, 'N solutions:')

    grid_levels_widget = create_select_multiple(
        options=[16, 32, 64, 128, 256],
        value=(16, 32, 64),
        description='Grid levels:',
    )

    n_sf_widget = create_int_slider(4, 2, 6, 1, 'SF degree:')

    vel_scale_widget = create_float_log_slider(
        value=1e5, min=2, max=6, step=0.5,
        description='Vel scale:',
        readout_format='.2e',
    )

    latent_dim_widget = create_int_slider(32, 8, 128, 8, 'Latent dim:')

    epochs_widget = create_int_slider(100, 10, 500, 10, 'Epochs:')

    batch_size_widget = create_int_slider(32, 8, 128, 8, 'Batch size:')

    seed_widget = create_int_input(12345, 'Seed:')

    return (
        n_sol_widget,
        grid_levels_widget,
        n_sf_widget,
        vel_scale_widget,
        latent_dim_widget,
        epochs_widget,
        batch_size_widget,
        seed_widget,
    )


def create_execution_mode_dropdown():
    """
    Creates a dropdown widget for selecting the execution mode.

    Returns:
        A Dropdown widget for selecting the execution mode.
    """
    execution_mode_dropdown = create_dropdown(
        [
            'Train Autoencoders',
            'Align Latents',
            'Evaluate',
            'Generate Data',
            'No Run',
        ],
        'No Run',
        'Execution Mode:',
    )

    def handle_dropdown_change(change):
        """
        Handler for dropdown value change. Outputs mode-specific configurations
        for demonstration.
        """
        config_map = {
            'Train Autoencoders': 'Train autoencoder models for each modality',
            'Align Latents': 'Align latent spaces across modalities',
            'Evaluate': 'Evaluate reconstruction and alignment errors',
            'Generate Data': 'Generate PDE solution data',
            'No Run': 'Skip runner, go to analysis of existing results',
        }
        custom_variable = config_map.get(change.new, 'No valid option selected')
        print(custom_variable + ' # For demonstration purposes')

    execution_mode_dropdown.observe(handle_dropdown_change, names='value')
    return execution_mode_dropdown


def get_args_from_widgets(
    n_sol_widget,
    grid_levels_widget,
    n_sf_widget,
    vel_scale_widget,
    latent_dim_widget,
    epochs_widget,
    batch_size_widget,
    seed_widget,
):
    """
    Convert individual widget values to an argparse.Namespace.

    Args:
        n_sol_widget: Number of solutions IntSlider widget.
        grid_levels_widget: Grid levels SelectMultiple widget.
        n_sf_widget: Streamfunction polynomial degree IntSlider widget.
        vel_scale_widget: Velocity scale FloatLogSlider widget.
        latent_dim_widget: Latent dimension IntSlider widget.
        epochs_widget: Epochs IntSlider widget.
        batch_size_widget: Batch size IntSlider widget.
        seed_widget: Random seed IntText widget.

    Returns:
        argparse.Namespace with all experiment parameters.
    """
    args = argparse.Namespace(
        n_sol=n_sol_widget.value,
        levels=list(grid_levels_widget.value),
        n_sf=n_sf_widget.value,
        vel_scale=vel_scale_widget.value,
        latent_dim=latent_dim_widget.value,
        epochs=epochs_widget.value,
        batch_size=batch_size_widget.value,
        seed=seed_widget.value,
    )

    return args


def display_widgets(widgets_tuple, exec_widget=None):
    """
    Display all widgets in a VBox layout.

    Args:
        widgets_tuple: Tuple of widgets returned by build_widgets.
        exec_widget: Optional execution mode widget to append.

    Returns:
        A VBox containing all widgets.
    """
    items = list(widgets_tuple)
    if exec_widget is not None:
        items.append(exec_widget)
    box = widgets.VBox(items)
    return box
