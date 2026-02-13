"""Register Docling plugins."""

from langchain_docling.picture_description import PictureDescriptionLangChainModel


def picture_description():
    """Picture description plugins."""
    return {
        "picture_description": [
            PictureDescriptionLangChainModel,
        ]
    }
