import pkg_resources

SRC_RESOURCES_DIR = pkg_resources.resource_filename("summarizer", "resources")

TEXT_FIELD = "text"
"""A long text or chapter text."""

GOLD_SUM_FIELD = "summary"
"""Gold summary."""

MODEL_SUM_FIELD = "system_sum"
"""Model predicted summary."""

