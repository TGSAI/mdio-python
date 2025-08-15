"""Unit tests for concrete seismic dataset template implementations."""

# Import all concrete template classes
from tests.unit.v1.helpers import validate_variable
from tests.unit.v1.templates.test_seismic_2d_poststack import _UNIT_METER

from mdio.schemas.dtype import ScalarType
from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate
from mdio.schemas.v1.templates.seismic_2d_poststack import Seismic2DPostStackTemplate
from mdio.schemas.v1.templates.seismic_3d_poststack import Seismic3DPostStackTemplate
from mdio.schemas.v1.templates.seismic_3d_prestack_cdp import Seismic3DPreStackCDPTemplate
from mdio.schemas.v1.templates.seismic_3d_prestack_shot import Seismic3DPreStackShotTemplate
from mdio.schemas.v1.templates.template_registry import TemplateRegistry


class TestSeismicTemplates:
    """Test cases for Seismic2DPostStackTemplate."""

    def test_custom_data_variable_name(self) -> None:
        """Test get_data_variable_name with custom names."""

        # Define a template with a custom data variable name 'velocity'
        class Velocity2DPostStackTemplate(Seismic2DPostStackTemplate):
            def __init__(self, domain: str):
                super().__init__(domain=domain)

            @property
            def _trace_variable_name(self) -> str:
                return "velocity"

            @property
            def _name(self) -> str:
                return f"Velocity2D{self._trace_domain.capitalize()}"

        t = Velocity2DPostStackTemplate("depth")
        assert t.name == "Velocity2DDepth"
        assert t.trace_variable_name == "velocity"

        dataset = t.build_dataset("Velocity 2D Depth Line 001", sizes=[2048, 4096], horizontal_coord_unit=_UNIT_METER)

        # Verify velocity variable
        validate_variable(
            dataset,
            name="velocity",
            dims=[("cdp", 2048), ("depth", 4096)],
            coords=["cdp_x", "cdp_y"],
            dtype=ScalarType.FLOAT32,
        )

    def test_get_name_time(self) -> None:
        """Test get_name with domain."""
        time_template = Seismic2DPostStackTemplate("time")
        dpth_template = Seismic2DPostStackTemplate("depth")

        assert time_template.name == "PostStack2DTime"
        assert dpth_template.name == "PostStack2DDepth"

        time_template = Seismic3DPostStackTemplate("time")
        dpth_template = Seismic3DPostStackTemplate("depth")

        assert time_template.name == "PostStack3DTime"
        assert dpth_template.name == "PostStack3DDepth"

        time_template = Seismic3DPreStackCDPTemplate("time")
        dpth_template = Seismic3DPreStackCDPTemplate("depth")

        assert time_template.name == "PreStackCdpGathers3DTime"
        assert dpth_template.name == "PreStackCdpGathers3DDepth"

        time_template = Seismic3DPreStackShotTemplate("time")
        dpth_template = Seismic3DPreStackShotTemplate("depth")

        assert time_template.name == "PreStackShotGathers3DTime"
        assert dpth_template.name == "PreStackShotGathers3DDepth"

    def test_all_templates_inherit_from_abstract(self) -> None:
        """Test that all concrete templates inherit from AbstractDatasetTemplate."""
        registry = TemplateRegistry()
        template_names = registry.list_all_templates()

        for template_name in template_names:
            template = registry.get(template_name)
            assert isinstance(template, AbstractDatasetTemplate)
            # That each template has the required properties and methods
            assert hasattr(template, "name")
            assert hasattr(template, "trace_variable_name")
            assert hasattr(template, "trace_domain")
            assert hasattr(template, "dimension_names")
            assert hasattr(template, "coordinate_names")
            assert hasattr(template, "build_dataset")

        assert len(template_names) == len(set(template_names)), f"Duplicate template names found: {template_names}"
