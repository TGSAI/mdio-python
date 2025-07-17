"""Unit tests for concrete seismic dataset template implementations."""

# Import all concrete template classes
from mdio.schemas.v1.templates.abstract_dataset_template import AbstractDatasetTemplate
from mdio.schemas.v1.templates.seismic_2d_poststack import Seismic2DPostStackTemplate
from mdio.schemas.v1.templates.seismic_3d_poststack import Seismic3DPostStackTemplate
from mdio.schemas.v1.templates.seismic_3d_prestack_cdp import Seismic3DPreStackCDPTemplate
from mdio.schemas.v1.templates.seismic_3d_prestack_shot import Seismic3DPreStackShotTemplate


class TestSeismicTemplates:
    """Test cases for Seismic2DPostStackTemplate."""

    def test_get_name_time(self) -> None:
        """Test get_name with domain."""
        time_template = Seismic2DPostStackTemplate("time")
        dpth_template = Seismic2DPostStackTemplate("depth")

        assert time_template.get_name() == "PostStack2DTime"
        assert dpth_template.get_name() == "PostStack2DDepth"

        time_template = Seismic3DPostStackTemplate("time")
        dpth_template = Seismic3DPostStackTemplate("depth")

        assert time_template.get_name() == "PostStack3DTime"
        assert dpth_template.get_name() == "PostStack3DDepth"

        time_template = Seismic3DPreStackCDPTemplate("time")
        dpth_template = Seismic3DPreStackCDPTemplate("depth")

        assert time_template.get_name() == "PreStackCdpGathers3DTime"
        assert dpth_template.get_name() == "PreStackCdpGathers3DDepth"

        time_template = Seismic3DPreStackShotTemplate("time")
        dpth_template = Seismic3DPreStackShotTemplate("depth")

        assert time_template.get_name() == "PreStackShotGathers3DTime"
        assert dpth_template.get_name() == "PreStackShotGathers3DDepth"

    def test_all_templates_inherit_from_abstract(self) -> None:
        """Test that all concrete templates inherit from AbstractDatasetTemplate."""
        templates = [
            Seismic2DPostStackTemplate("time"),
            Seismic3DPostStackTemplate("time"),
            Seismic3DPreStackCDPTemplate("time"),
            Seismic3DPreStackShotTemplate("time"),
            Seismic2DPostStackTemplate("depth"),
            Seismic3DPostStackTemplate("depth"),
            Seismic3DPreStackCDPTemplate("depth"),
            Seismic3DPreStackShotTemplate("depth"),
        ]

        for template in templates:
            assert isinstance(template, AbstractDatasetTemplate)
            assert hasattr(template, "get_name")
            assert hasattr(template, "build_dataset")

        names = [template.get_name() for template in templates]
        assert len(names) == len(set(names)), f"Duplicate template names found: {names}"
