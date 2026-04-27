"""Google Sheets OAuth uploader for collaborative tasks.

Uses personal Google account OAuth instead of service account.
Creates fresh Google Sheets in your personal Drive.
"""

import io
import logging
import os
import pickle
import tempfile
from typing import Dict, Any

import requests
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

logger = logging.getLogger(__name__)

# Scopes needed for creating sheets/docs/slides and setting permissions
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/documents',
    'https://www.googleapis.com/auth/presentations',
    'https://www.googleapis.com/auth/drive.file'
]

_credentials_cache = None


def _get_oauth_credentials(
    client_secret_path: str = "oauth_client_secret.json",
    token_path: str = "oauth_token.pickle"
):
    """Get OAuth credentials, prompting for login if needed.

    Args:
        client_secret_path: Path to OAuth client secret JSON
        token_path: Path to save/load refresh token

    Returns:
        Authenticated credentials object
    """
    global _credentials_cache

    if _credentials_cache is not None:
        return _credentials_cache

    creds = None

    # Load saved token if it exists
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)

    # If no valid credentials, do OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("[oauth] Refreshing expired token")
            creds.refresh(Request())
        else:
            logger.info("[oauth] Starting OAuth flow - manual code entry")
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secret_path,
                SCOPES,
                redirect_uri='urn:ietf:wg:oauth:2.0:oob'
            )

            # Use console flow for remote servers
            auth_url, _ = flow.authorization_url(prompt='consent')

            logger.info("\n" + "="*70)
            logger.info("AUTHORIZATION REQUIRED")
            logger.info("="*70)
            logger.info("\n1. Open this URL in your browser:")
            logger.info("\n   %s\n", auth_url)
            logger.info("2. Authorize the application")
            logger.info("3. Copy the authorization code from the browser")
            logger.info("4. Paste it below\n")

            code = input("Enter authorization code: ").strip()
            flow.fetch_token(code=code)
            creds = flow.credentials

        # Save token for future runs
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
        logger.info("[oauth] Token saved to %s", token_path)

    _credentials_cache = creds
    return creds


def create_sheet_from_template_oauth(
    template_url: str,
    client_secret_path: str = "oauth_client_secret.json",
    token_path: str = "oauth_token.pickle",
    title: str = "OSWorld Collaborative Task Sheet"
) -> str:
    """Create a new Google Sheet from an .xlsx template using OAuth.

    Args:
        template_url: URL to download .xlsx template from (e.g., HuggingFace)
        client_secret_path: Path to OAuth client secret JSON
        token_path: Path to save/load OAuth refresh token
        title: Title for the new Google Sheet

    Returns:
        Shareable Google Sheets URL (anyone with link can edit)
    """
    logger.info("[sheets-oauth] Downloading template from %s", template_url)

    # Download template .xlsx
    response = requests.get(template_url, stream=True)
    response.raise_for_status()

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
        tmp_path = tmp.name

    try:
        # Get OAuth credentials
        creds = _get_oauth_credentials(client_secret_path, token_path)
        drive_service = build('drive', 'v3', credentials=creds)

        logger.info("[sheets-oauth] Uploading to Google Sheets as '%s'", title)

        # Upload .xlsx file and convert to Google Sheets format
        file_metadata = {
            'name': title,
            'mimeType': 'application/vnd.google-apps.spreadsheet'
        }

        with open(tmp_path, 'rb') as fh:
            media = MediaIoBaseUpload(
                io.BytesIO(fh.read()),
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                resumable=True
            )

            file = drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,webViewLink'
            ).execute()

        file_id = file.get('id')

        # Set permissions to "anyone with link can edit"
        logger.info("[sheets-oauth] Setting permissions (anyone with link can edit)")
        drive_service.permissions().create(
            fileId=file_id,
            body={
                'type': 'anyone',
                'role': 'writer'
            }
        ).execute()

        # Get shareable URL
        sheet_url = file.get('webViewLink')
        # Convert view link to edit link
        sheet_url = sheet_url.replace('/view', '/edit')

        logger.info("[sheets-oauth] Created sheet: %s", sheet_url)
        return sheet_url

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def create_doc_from_template_oauth(
    template_url: str,
    client_secret_path: str = "oauth_client_secret.json",
    token_path: str = "oauth_token.pickle",
    title: str = "OSWorld Collaborative Task Doc"
) -> str:
    """Create a new Google Doc from a .docx template using OAuth.

    Args:
        template_url: URL to download .docx template from (e.g., HuggingFace)
        client_secret_path: Path to OAuth client secret JSON
        token_path: Path to save/load OAuth refresh token
        title: Title for the new Google Doc

    Returns:
        Shareable Google Docs URL (anyone with link can edit)
    """
    logger.info("[docs-oauth] Downloading template from %s", template_url)

    # Download template .docx
    response = requests.get(template_url, stream=True)
    response.raise_for_status()

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
        tmp_path = tmp.name

    try:
        # Get OAuth credentials
        creds = _get_oauth_credentials(client_secret_path, token_path)
        drive_service = build('drive', 'v3', credentials=creds)

        logger.info("[docs-oauth] Uploading to Google Docs as '%s'", title)

        # Upload .docx file and convert to Google Docs format
        file_metadata = {
            'name': title,
            'mimeType': 'application/vnd.google-apps.document'
        }

        with open(tmp_path, 'rb') as fh:
            media = MediaIoBaseUpload(
                io.BytesIO(fh.read()),
                mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                resumable=True
            )

            file = drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,webViewLink'
            ).execute()

        file_id = file.get('id')

        # Set permissions to "anyone with link can edit"
        logger.info("[docs-oauth] Setting permissions (anyone with link can edit)")
        drive_service.permissions().create(
            fileId=file_id,
            body={
                'type': 'anyone',
                'role': 'writer'
            }
        ).execute()

        # Get shareable URL
        doc_url = file.get('webViewLink')
        # Convert view link to edit link
        doc_url = doc_url.replace('/view', '/edit')

        logger.info("[docs-oauth] Created doc: %s", doc_url)
        return doc_url

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def get_sheet_id_from_url(url: str) -> str:
    """Extract sheet/doc ID from Google Sheets/Docs URL."""
    # URL format: https://docs.google.com/spreadsheets/d/{id}/edit...
    # or: https://docs.google.com/document/d/{id}/edit...
    parts = url.split('/d/')
    if len(parts) < 2:
        raise ValueError(f"Invalid Google Workspace URL: {url}")
    doc_id = parts[1].split('/')[0]
    return doc_id


def reset_sheet_from_template(
    sheet_url: str,
    template_url: str,
    client_secret_path: str = "oauth_client_secret.json",
    token_path: str = "oauth_token.pickle",
) -> bool:
    """Reset an existing Google Sheet to template state.

    Replaces the entire content of an existing sheet with fresh template data.
    The URL stays the same, permissions persist.

    Args:
        sheet_url: URL of existing Google Sheet to reset
        template_url: URL to download .xlsx template from
        client_secret_path: Path to OAuth client secret JSON
        token_path: Path to save/load OAuth refresh token

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("[sheets-oauth] Resetting sheet to template state")
    logger.info("[sheets-oauth]   Sheet URL: %s", sheet_url)
    logger.info("[sheets-oauth]   Template: %s", template_url)

    # Extract sheet ID from URL
    sheet_id = get_sheet_id_from_url(sheet_url)

    # Download template .xlsx
    logger.info("[sheets-oauth] Downloading template")
    response = requests.get(template_url, stream=True)
    response.raise_for_status()

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
        tmp_path = tmp.name

    try:
        # Get OAuth credentials
        creds = _get_oauth_credentials(client_secret_path, token_path)
        drive_service = build('drive', 'v3', credentials=creds)

        logger.info("[sheets-oauth] Replacing sheet content via Drive API")

        # Replace the entire file content
        # This preserves the file ID (and thus the URL) but replaces all content
        with open(tmp_path, 'rb') as fh:
            media = MediaIoBaseUpload(
                io.BytesIO(fh.read()),
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                resumable=True
            )

            drive_service.files().update(
                fileId=sheet_id,
                media_body=media,
            ).execute()

        logger.info("[sheets-oauth] ✓ Sheet reset successful")
        return True

    except Exception as e:
        logger.error(f"[sheets-oauth] Failed to reset sheet: {e}")
        return False

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
def create_slide_from_template_oauth(
    template_url: str,
    client_secret_path: str = "oauth_client_secret.json",
    token_path: str = "oauth_token.pickle",
    title: str = "OSWorld Collaborative Task Slides"
) -> str:
    """Create a new Google Slides from a .pptx template using OAuth.

    Args:
        template_url: URL to download .pptx template from (e.g., HuggingFace)
        client_secret_path: Path to OAuth client secret JSON
        token_path: Path to save/load OAuth refresh token
        title: Title for the new Google Slides

    Returns:
        Shareable Google Slides URL (anyone with link can edit)
    """
    logger.info("[slides-oauth] Downloading template from %s", template_url)

    # Download template .pptx
    response = requests.get(template_url, stream=True)
    response.raise_for_status()

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pptx', delete=False) as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
        tmp_path = tmp.name

    try:
        # Get OAuth credentials
        creds = _get_oauth_credentials(client_secret_path, token_path)
        drive_service = build('drive', 'v3', credentials=creds)

        logger.info("[slides-oauth] Uploading to Google Slides as '%s'", title)

        # Upload .pptx file and convert to Google Slides format
        file_metadata = {
            'name': title,
            'mimeType': 'application/vnd.google-apps.presentation'
        }

        # Read and upload the file
        media = MediaIoBaseUpload(
            open(tmp_path, 'rb'),
            mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation',
            resumable=True
        )

        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink'
        ).execute()

        slide_id = file.get('id')
        logger.info("[slides-oauth] ✓ Slides created: %s", slide_id)

        # Set permissions: anyone with link can edit
        permission = {
            'type': 'anyone',
            'role': 'writer'
        }
        drive_service.permissions().create(
            fileId=slide_id,
            body=permission
        ).execute()

        logger.info("[slides-oauth] ✓ Permissions set to 'anyone can edit'")

        # Get shareable link
        file_meta = drive_service.files().get(
            fileId=slide_id,
            fields='webViewLink'
        ).execute()

        url = file_meta.get('webViewLink')
        logger.info("[slides-oauth] ✓ Shareable URL: %s", url)

        return url

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
