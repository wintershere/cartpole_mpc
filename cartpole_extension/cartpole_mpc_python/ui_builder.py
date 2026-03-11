import omni.timeline
import omni.ui as ui
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import create_new_stage, get_current_stage
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.gui.components.element_wrappers import CollapsableFrame, StateButton
from isaacsim.gui.components.ui_utils import get_style
from omni.usd import StageEventType
from pxr import Sdf, UsdLux

from .scenario import CartpoleMPCScript


class UIBuilder:
    def __init__(self):
        self.frames = []
        self.wrapped_ui_elements = []
        self._timeline = omni.timeline.get_timeline_interface()
        self._on_init()

    ###################################################################################
    #           The Functions Below Are Called Automatically By extension.py
    ###################################################################################

    def on_menu_callback(self):
        pass

    def on_timeline_event(self, event):
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            if hasattr(self, "_scenario_state_btn") and self._scenario_state_btn is not None:
                self._scenario_state_btn.reset()
                self._scenario_state_btn.enabled = False

    def on_physics_step(self, step: float):
        pass

    def on_stage_event(self, event):
        if event.type == int(StageEventType.OPENED):
            self._reset_extension()

    def cleanup(self):
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()

    def build_ui(self):
        # ── World Controls ────────────────────────────────────────────────────
        world_controls_frame = CollapsableFrame("World Controls", collapsed=False)
        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._load_btn = LoadButton(
                    "Load Button", "LOAD",
                    setup_scene_fn=self._setup_scene,
                    setup_post_load_fn=self._setup_scenario,
                )
                self._load_btn.set_world_settings(
                    physics_dt=1 / 25.0, rendering_dt=1 / 60.0
                )
                self.wrapped_ui_elements.append(self._load_btn)

                self._reset_btn = ResetButton(
                    "Reset Button", "RESET",
                    pre_reset_fn=None,
                    post_reset_fn=self._on_post_reset_btn,
                )
                self._reset_btn.enabled = False
                self.wrapped_ui_elements.append(self._reset_btn)

        # ── MPC Config ────────────────────────────────────────────────────────
        config_frame = CollapsableFrame("MPC Config", collapsed=False)
        with config_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                with ui.HStack(spacing=10):
                    ui.Label("RTI scheme", width=100)
                    self._rti_cb = ui.CheckBox()
                    self._rti_cb.model.set_value(True)

                with ui.HStack(spacing=10):
                    ui.Label("Add noise", width=100)
                    self._noise_cb = ui.CheckBox()
                    self._noise_cb.model.set_value(True)

                with ui.HStack(spacing=10):
                    ui.Label("Add impulse", width=100)
                    self._impulse_cb = ui.CheckBox()
                    self._impulse_cb.model.set_value(True)

        # ── Run Scenario ──────────────────────────────────────────────────────
        run_scenario_frame = CollapsableFrame("Run Scenario")
        with run_scenario_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._scenario_state_btn = StateButton(
                    "Run MPC",
                    "RUN",
                    "STOP",
                    on_a_click_fn=self._on_run_scenario_a_text,
                    on_b_click_fn=self._on_run_scenario_b_text,
                    physics_callback_fn=self._update_scenario,
                )
                self._scenario_state_btn.enabled = False
                self.wrapped_ui_elements.append(self._scenario_state_btn)

    ######################################################################################
    # Functions Below This Point Support The MPC Scenario
    ######################################################################################

    def _on_init(self):
        self._scenario = CartpoleMPCScript()

    def _add_light_to_stage(self):
        sphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        SingleXFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])

    def _setup_scene(self):
        create_new_stage()
        self._add_light_to_stage()

        # Pass UI config to scenario before setup() is called
        self._scenario.use_rti     = self._rti_cb.model.get_value_as_bool()
        self._scenario.add_noise   = self._noise_cb.model.get_value_as_bool()
        self._scenario.add_impulse = self._impulse_cb.model.get_value_as_bool()

        loaded_objects = self._scenario.load_example_assets()

        world = World.instance()
        for obj in loaded_objects:
            world.scene.add(obj)

    def _setup_scenario(self):
        self._scenario.setup()

        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._reset_btn.enabled = True

    def _on_post_reset_btn(self):
        self._scenario.reset()
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True

    def _update_scenario(self, step: float):
        done = self._scenario.update(step)
        if done:
            self._scenario_state_btn.enabled = False

    def _on_run_scenario_a_text(self):
        self._timeline.play()

    def _on_run_scenario_b_text(self):
        self._timeline.pause()

    def _reset_extension(self):
        self._on_init()
        self._reset_ui()

    def _reset_ui(self):
        if hasattr(self, "_scenario_state_btn") and self._scenario_state_btn is not None:
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False
        if hasattr(self, "_reset_btn") and self._reset_btn is not None:
            self._reset_btn.enabled = False
