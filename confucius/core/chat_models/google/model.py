# pyre-strict


def get_model(model: str) -> str:
    if model.startswith("gemini-2-5-pro"):
        return "gemini-2.5-pro"
    if model.startswith("gemini-2-5-flash"):
        return "gemini-2.5-flash"

    return model
