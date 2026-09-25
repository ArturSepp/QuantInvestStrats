"""Render portable Insight and Pitfall blockquotes as admonitions.

Methodology articles mark a practical insight or a common error as an ordinary Markdown
blockquote whose first word is bold, for example ``> **Insight.** ...``. GitHub and VS Code show
a readable quotation; this transform turns the same source into a styled admonition in Sphinx,
so the articles need no renderer-specific directive. Any other blockquote is left unchanged.
"""

from typing import Any, Dict

from docutils import nodes
from sphinx.transforms import SphinxTransform


# label written in the source -> (admonition title, theme class that sets its colour)
CALLOUTS: Dict[str, tuple] = {
    'Insight.': ('Insight', 'tip'),
    'Pitfall.': ('Pitfall', 'warning'),
}


class CalloutTransform(SphinxTransform):
    """Replace a labelled blockquote with a titled admonition."""

    default_priority = 500

    def apply(self, **kwargs: Any) -> None:
        """Rewrite every blockquote whose first paragraph starts with a callout label.

        Args:
            **kwargs: unused transform arguments supplied by Sphinx
        """
        for quote in list(self.document.findall(nodes.block_quote)):
            if not quote.children or not isinstance(quote.children[0], nodes.paragraph):
                continue
            paragraph = quote.children[0]
            # the Sphinx reader can leave an empty text node before the bold label
            while (paragraph.children and isinstance(paragraph.children[0], nodes.Text)
                   and not paragraph.children[0].astext().strip()):
                paragraph.remove(paragraph.children[0])
            if not paragraph.children or not isinstance(paragraph.children[0], nodes.strong):
                continue
            label = paragraph.children[0].astext().strip()
            if label not in CALLOUTS:
                continue
            title, theme_class = CALLOUTS[label]
            paragraph.remove(paragraph.children[0])
            if paragraph.children and isinstance(paragraph.children[0], nodes.Text):
                stripped = paragraph.children[0].astext().lstrip()
                paragraph.replace(paragraph.children[0], nodes.Text(stripped))
            admonition = nodes.admonition(
                '', nodes.title('', title), *quote.children,
                classes=[theme_class, f'qis-{title.lower()}'])
            admonition.source, admonition.line = quote.source, quote.line
            quote.replace_self(admonition)


def setup(app: Any) -> dict:
    """Register the callout transform.

    Args:
        app: Sphinx application.

    Returns:
        Extension metadata.
    """
    app.add_transform(CalloutTransform)
    return {'version': '1', 'parallel_read_safe': True, 'parallel_write_safe': True}
