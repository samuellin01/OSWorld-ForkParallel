import json
import logging
import os
import os.path
import shutil
import sqlite3
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timedelta
from typing import Any, Union, Optional
from typing import Dict, List

import requests
from playwright.sync_api import sync_playwright, TimeoutError
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive, GoogleDriveFile, GoogleDriveFileList
from requests_toolbelt.multipart.encoder import MultipartEncoder

from desktop_env.controllers.python import PythonController
from desktop_env.evaluators.metrics.utils import compare_urls
from desktop_env.providers.aws.proxy_pool import get_global_proxy_pool, init_proxy_pool, ProxyInfo

import dotenv
# Load environment variables from .env file
dotenv.load_dotenv()


PROXY_CONFIG_FILE = os.getenv("PROXY_CONFIG_FILE", "evaluation_examples/settings/proxy/dataimpulse.json")  # Default proxy config file

logger = logging.getLogger("desktopenv.setup")

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

init_proxy_pool(PROXY_CONFIG_FILE)  # initialize the global proxy pool

MAX_RETRIES = 20

class SetupController:
    def __init__(self, vm_ip: str, server_port: int = 5000, chromium_port: int = 9222, vlc_port: int = 8080, cache_dir: str = "cache", client_password: str = "", screen_width: int = 1920, screen_height: int = 1080):
        self.vm_ip: str = vm_ip
        self.server_port: int = server_port
        self.chromium_port: int = chromium_port
        self.vlc_port: int = vlc_port
        self.http_server: str = f"http://{vm_ip}:{server_port}"
        self.http_server_setup_root: str = f"http://{vm_ip}:{server_port}/setup"
        self.cache_dir: str = cache_dir
        self.use_proxy: bool = False
        self.client_password: str = client_password
        self.screen_width: int = screen_width
        self.screen_height: int = screen_height

    def reset_cache_dir(self, cache_dir: str):
        self.cache_dir = cache_dir

    def setup(self, config: List[Dict[str, Any]], use_proxy: bool = False)-> bool:
        """
        Args:
            config (List[Dict[str, Any]]): list of dict like {str: Any}. each
              config dict has the structure like
                {
                    "type": str, corresponding to the `_{:}_setup` methods of
                      this class
                    "parameters": dict like {str, Any} providing the keyword
                      parameters
                }
        """  
        self.use_proxy = use_proxy
        # make sure connection can be established
        logger.info(f"try to connect {self.http_server}")
        retry = 0
        while retry < MAX_RETRIES:
            try:
                resp = requests.get(self.http_server + "/terminal", timeout=10)
                if resp.status_code < 500:
                    break
                logger.info(f"Server returned {resp.status_code}, not ready yet. retry: {retry+1}/{MAX_RETRIES}")
            except:
                logger.info(f"retry: {retry+1}/{MAX_RETRIES}")
            time.sleep(5)
            retry += 1

            if retry == MAX_RETRIES:
                return False
                

        for i, cfg in enumerate(config):
            config_type: str = cfg["type"]
            parameters: Dict[str, Any] = cfg["parameters"]

            # Assumes all the setup the functions should follow this name
            # protocol
            setup_function: str = "_{:}_setup".format(config_type)
            assert hasattr(self, setup_function), f'Setup controller cannot find init function {setup_function}'
            
            try:
                logger.info(f"Executing setup step {i+1}/{len(config)}: {setup_function}")
                logger.debug(f"Setup parameters: {parameters}")
                getattr(self, setup_function)(**parameters)
                logger.info(f"SETUP COMPLETED: {setup_function}({str(parameters)})")
            except Exception as e:
                logger.error(f"SETUP FAILED at step {i+1}/{len(config)}: {setup_function}({str(parameters)})")
                logger.error(f"Error details: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise Exception(f"Setup step {i+1} failed: {setup_function} - {e}") from e
        
        return True

    def _download_setup(self, files: List[Dict[str, str]]):
        """
        Args:
            files (List[Dict[str, str]]): files to download. lisf of dict like
              {
                "url": str, the url to download
                "path": str, the path on the VM to store the downloaded file
              }
        """
        for f in files:
            url: str = f["url"]
            path: str = f["path"]
            cache_path: str = os.path.join(self.cache_dir, "{:}_{:}".format(
                uuid.uuid5(uuid.NAMESPACE_URL, url),
                os.path.basename(path)))
            if not url or not path:
                raise Exception(f"Setup Download - Invalid URL ({url}) or path ({path}).")

            if not os.path.exists(cache_path):
                logger.info(f"Cache file not found, downloading from {url} to {cache_path}")
                max_retries = 3
                downloaded = False
                e = None
                for i in range(max_retries):
                    try:
                        logger.info(f"Download attempt {i+1}/{max_retries} for {url}")
                        response = requests.get(url, stream=True, timeout=300)  # Add 5 minute timeout
                        response.raise_for_status()
                        
                        # Get file size if available
                        total_size = int(response.headers.get('content-length', 0))
                        if total_size > 0:
                            logger.info(f"File size: {total_size / (1024*1024):.2f} MB")

                        downloaded_size = 0
                        with open(cache_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    downloaded_size += len(chunk)
                                    if total_size > 0 and downloaded_size % (1024*1024) == 0:  # Log every MB
                                        progress = (downloaded_size / total_size) * 100
                                        logger.info(f"Download progress: {progress:.1f}%")
                        
                        logger.info(f"File downloaded successfully to {cache_path} ({downloaded_size / (1024*1024):.2f} MB)")
                        downloaded = True
                        break

                    except requests.RequestException as e:
                        logger.error(
                            f"Failed to download {url} caused by {e}. Retrying... ({max_retries - i - 1} attempts left)")
                        # Clean up partial download
                        if os.path.exists(cache_path):
                            os.remove(cache_path)
                if not downloaded:
                    raise requests.RequestException(f"Failed to download {url}. No retries left.")

            # Upload to VM with retry for transient errors
            upload_max_retries = 5
            upload_backoff = 5
            upload_last_error = None

            for upload_attempt in range(upload_max_retries):
                try:
                    with open(cache_path, "rb") as file_data:
                        form = MultipartEncoder({
                            "file_path": path,
                            "file_data": (os.path.basename(path), file_data)
                        })
                        headers = {"Content-Type": form.content_type}
                        logger.debug(form.content_type)

                        logger.info(f"Uploading {os.path.basename(path)} to VM at {path} (attempt {upload_attempt + 1}/{upload_max_retries})")
                        logger.debug("REQUEST ADDRESS: %s", self.http_server + "/setup" + "/upload")
                        response = requests.post(
                            self.http_server + "/setup" + "/upload",
                            headers=headers,
                            data=form,
                            timeout=(10, 600),
                        )
                    if response.status_code == 200:
                        logger.info(f"File uploaded successfully: {path}")
                        logger.debug("Upload response: %s", response.text)
                        break
                    elif response.status_code in (502, 503, 504):
                        upload_last_error = f"Status {response.status_code}: {response.text}"
                        logger.warning(
                            "Upload attempt %d/%d got %d, retrying in %ds...",
                            upload_attempt + 1, upload_max_retries, response.status_code, upload_backoff,
                        )
                    else:
                        logger.error(f"Failed to upload file {path}. Status code: {response.status_code}, Response: {response.text}")
                        raise requests.RequestException(f"Upload failed with status {response.status_code}")
                except requests.exceptions.RequestException as e:
                    upload_last_error = str(e)
                    logger.warning(
                        "Upload attempt %d/%d failed for %s: %s. Retrying in %ds...",
                        upload_attempt + 1, upload_max_retries, path, e, upload_backoff,
                    )

                if upload_attempt < upload_max_retries - 1:
                    time.sleep(upload_backoff)
                    upload_backoff = min(upload_backoff * 2, 30)

            if upload_last_error is not None:
                raise requests.RequestException(
                    f"All {upload_max_retries} upload attempts failed for {path}. Last error: {upload_last_error}"
                )

    def _upload_file_setup(self, files: List[Dict[str, str]]):
        """
        Args:
            files (List[Dict[str, str]]): files to download. lisf of dict like
              {
                "local_path": str, the local path to the file to upload
                "path": str, the path on the VM to store the downloaded file
              }
        """
        for f in files:
            local_path: str = f["local_path"]
            path: str = f["path"]

            if not os.path.exists(local_path):
                raise Exception(f"Setup Upload - Invalid local path ({local_path}).")

            file_size = None
            try:
                file_size = os.path.getsize(local_path)
            except Exception:
                pass

            max_retries = 3
            last_error: Optional[Exception] = None

            for attempt in range(max_retries):
                try:
                    logger.info(
                        f"Uploading {os.path.basename(local_path)}{f' ({file_size} bytes)' if file_size is not None else ''} "
                        f"to VM at {path} (attempt {attempt + 1}/{max_retries})"
                    )
                    logger.debug("REQUEST ADDRESS: %s", self.http_server + "/setup" + "/upload")

                    # Open the file inside each attempt to ensure fresh stream position
                    with open(local_path, "rb") as fp:
                        form = MultipartEncoder({
                            "file_path": path,
                            "file_data": (os.path.basename(path), fp)
                        })
                        headers = {"Content-Type": form.content_type}
                        logger.debug(form.content_type)

                        # Explicit connect/read timeout to avoid hanging forever
                        response = requests.post(
                            self.http_server + "/setup" + "/upload",
                            headers=headers,
                            data=form,
                            timeout=(10, 600)
                        )

                        if response.status_code == 200:
                            logger.info(f"File uploaded successfully: {path}")
                            logger.debug("Upload response: %s", response.text)
                            last_error = None
                            break
                        else:
                            msg = f"Failed to upload file {path}. Status code: {response.status_code}, Response: {response.text}"
                            logger.error(msg)
                            last_error = requests.RequestException(msg)

                except requests.exceptions.RequestException as e:
                    last_error = e
                    logger.error(f"Upload attempt {attempt + 1} failed for {path}: {e}")

                # Exponential backoff between retries
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

            if last_error is not None:
                raise last_error

    def _change_wallpaper_setup(self, path: str):
        if not path:
            raise Exception(f"Setup Wallpaper - Invalid path ({path}).")

        payload = json.dumps({"path": path})
        headers = {
            'Content-Type': 'application/json'
        }

        # send request to server to change wallpaper
        try:
            response = requests.post(self.http_server + "/setup" + "/change_wallpaper", headers=headers, data=payload)
            if response.status_code == 200:
                logger.info("Command executed successfully: %s", response.text)
            else:
                logger.error("Failed to change wallpaper. Status code: %s", response.text)
        except requests.exceptions.RequestException as e:
            logger.error("An error occurred while trying to send the request: %s", e)

    def _tidy_desktop_setup(self, **config):
        raise NotImplementedError()

    def _open_setup(self, path: str):
        if not path:
            raise Exception(f"Setup Open - Invalid path ({path}).")

        payload = json.dumps({"path": path})
        headers = {
            'Content-Type': 'application/json'
        }

        # send request to server to open file
        try:
            # The server-side call is now blocking and can take time.
            # We set a timeout that is slightly longer than the server's timeout (1800s).
            response = requests.post(self.http_server + "/setup" + "/open_file", headers=headers, data=payload, timeout=1810)
            response.raise_for_status()  # This will raise an exception for 4xx and 5xx status codes
            logger.info("Command executed successfully: %s", response.text)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to open file '{path}'. An error occurred while trying to send the request or the server responded with an error: {e}")
            raise Exception(f"Failed to open file '{path}'. An error occurred while trying to send the request or the server responded with an error: {e}") from e

    def _launch_setup(self, command: Union[str, List[str]], shell: bool = False):
        if not command:
            raise Exception("Empty command to launch.")

        if not shell and isinstance(command, str) and len(command.split()) > 1:
            logger.warning("Command should be a list of strings. Now it is a string. Will split it by space.")
            command = command.split()
            
        if command[0] == "google-chrome":
            if "--remote-allow-origins=*" not in command:
                command.append("--remote-allow-origins=*")
            if self.use_proxy:
                command.append("--proxy-server=http://127.0.0.1:18888")  # Use the proxy server set up by _proxy_setup

        payload = json.dumps({"command": command, "shell": shell})
        headers = {"Content-Type": "application/json"}

        max_retries = 5
        backoff = 5
        last_error = None

        for attempt in range(max_retries):
            try:
                logger.info("REQUEST ADDRESS: %s", self.http_server + "/setup" + "/launch")
                response = requests.post(
                    self.http_server + "/setup" + "/launch",
                    headers=headers,
                    data=payload,
                    timeout=(10, 300),
                )
                if response.status_code == 200:
                    logger.info("Command executed successfully: %s", response.text)
                    return
                elif response.status_code in (502, 503, 504):
                    last_error = f"Status {response.status_code}: {response.text}"
                    logger.warning(
                        "Launch attempt %d/%d got %d, retrying in %ds...",
                        attempt + 1, max_retries, response.status_code, backoff,
                    )
                else:
                    logger.error("Failed to launch application. Status code: %s", response.text)
                    return
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.warning(
                    "Launch attempt %d/%d failed: %s. Retrying in %ds...",
                    attempt + 1, max_retries, e, backoff,
                )

            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)

        logger.error("All %d launch attempts failed. Last error: %s", max_retries, last_error)

    def _execute_setup(
            self,
            command: List[str],
            stdout: str = "",
            stderr: str = "",
            shell: bool = False,
            until: Optional[Dict[str, Any]] = None
    ):
        if not command:
            raise Exception("Empty command to launch.")

        until: Dict[str, Any] = until or {}
        terminates: bool = False
        nb_failings = 0

        def replace_screen_env_in_command(command):
            password = self.client_password
            width = self.screen_width
            height = self.screen_height
            width_half = str(width // 2)
            height_half = str(height // 2)
            new_command_list = []
            new_command = ""
            if isinstance(command, str):
                new_command = command.replace("{CLIENT_PASSWORD}", password)
                new_command = new_command.replace("{SCREEN_WIDTH_HALF}", width_half)
                new_command = new_command.replace("{SCREEN_HEIGHT_HALF}", height_half)
                new_command = new_command.replace("{SCREEN_WIDTH}", str(width))
                new_command = new_command.replace("{SCREEN_HEIGHT}", str(height))
                return new_command
            else:
                for item in command:
                    item = item.replace("{CLIENT_PASSWORD}", password)
                    item = item.replace("{SCREEN_WIDTH_HALF}", width_half)
                    item = item.replace("{SCREEN_HEIGHT_HALF}", height_half)
                    item = item.replace("{SCREEN_WIDTH}", str(width))
                    item = item.replace("{SCREEN_HEIGHT}", str(height))
                    new_command_list.append(item)
                return new_command_list
        command = replace_screen_env_in_command(command)
        payload = json.dumps({"command": command, "shell": shell})
        headers = {"Content-Type": "application/json"}

        while not terminates:
            try:
                response = requests.post(self.http_server + "/setup" + "/execute", headers=headers, data=payload)
                if response.status_code == 200:
                    results: Dict[str, str] = response.json()
                    if stdout:
                        with open(os.path.join(self.cache_dir, stdout), "w") as f:
                            f.write(results["output"])
                    if stderr:
                        with open(os.path.join(self.cache_dir, stderr), "w") as f:
                            f.write(results["error"])
                    logger.info("Command executed successfully: %s -> %s"
                                , " ".join(command) if isinstance(command, list) else command
                                , response.text
                                )
                else:
                    logger.error("Failed to launch application. Status code: %s", response.text)
                    results = None
                    nb_failings += 1
            except requests.exceptions.RequestException as e:
                logger.error("An error occurred while trying to send the request: %s", e)
                traceback.print_exc()

                results = None
                nb_failings += 1

            if len(until) == 0:
                terminates = True
            elif results is not None:
                terminates = "returncode" in until and results["returncode"] == until["returncode"] \
                             or "stdout" in until and until["stdout"] in results["output"] \
                             or "stderr" in until and until["stderr"] in results["error"]
            terminates = terminates or nb_failings >= 5
            if not terminates:
                time.sleep(0.3)

    def _execute_with_verification_setup(
            self,
            command: List[str],
            verification: Dict[str, Any] = None,
            max_wait_time: int = 10,
            check_interval: float = 1.0,
            shell: bool = False
    ):
        """Execute command with verification of results
        
        Args:
            command: Command to execute
            verification: Dict with verification criteria:
                - window_exists: Check if window with this name exists
                - command_success: Execute this command and check if it succeeds
            max_wait_time: Maximum time to wait for verification
            check_interval: Time between verification checks
            shell: Whether to use shell
        """
        if not command:
            raise Exception("Empty command to launch.")

        verification = verification or {}
        
        payload = json.dumps({
            "command": command, 
            "shell": shell,
            "verification": verification,
            "max_wait_time": max_wait_time,
            "check_interval": check_interval
        })
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.http_server + "/setup" + "/execute_with_verification", 
                                   headers=headers, data=payload, timeout=max_wait_time + 10)
            if response.status_code == 200:
                result = response.json()
                logger.info("Command executed and verified successfully: %s -> %s"
                            , " ".join(command) if isinstance(command, list) else command
                            , response.text
                            )
                return result
            else:
                logger.error("Failed to execute with verification. Status code: %s", response.text)
                raise Exception(f"Command verification failed: {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error("An error occurred while trying to send the request: %s", e)
            traceback.print_exc()
            raise Exception(f"Request failed: {e}")

    def _command_setup(self, command: List[str], **kwargs):
        self._execute_setup(command, **kwargs)

    def _sleep_setup(self, seconds: float):
        time.sleep(seconds)

    def _act_setup(self, action_seq: List[Union[Dict[str, Any], str]]):
        # TODO
        raise NotImplementedError()

    def _replay_setup(self, trajectory: str):
        """
        Args:
            trajectory (str): path to the replay trajectory file
        """

        # TODO
        raise NotImplementedError()

    def _activate_window_setup(self, window_name: str, strict: bool = False, by_class: bool = False):
        if not window_name:
            raise Exception(f"Setup Open - Invalid path ({window_name}).")

        payload = json.dumps({"window_name": window_name, "strict": strict, "by_class": by_class})
        headers = {
            'Content-Type': 'application/json'
        }

        # send request to server to open file
        try:
            response = requests.post(self.http_server + "/setup" + "/activate_window", headers=headers, data=payload)
            if response.status_code == 200:
                logger.info("Command executed successfully: %s", response.text)
            else:
                logger.error(f"Failed to activate window {window_name}. Status code: %s", response.text)
        except requests.exceptions.RequestException as e:
            logger.error("An error occurred while trying to send the request: %s", e)

    def _close_window_setup(self, window_name: str, strict: bool = False, by_class: bool = False):
        if not window_name:
            raise Exception(f"Setup Open - Invalid path ({window_name}).")

        payload = json.dumps({"window_name": window_name, "strict": strict, "by_class": by_class})
        headers = {
            'Content-Type': 'application/json'
        }

        # send request to server to open file
        try:
            response = requests.post(self.http_server + "/setup" + "/close_window", headers=headers, data=payload)
            if response.status_code == 200:
                logger.info("Command executed successfully: %s", response.text)
            else:
                logger.error(f"Failed to close window {window_name}. Status code: %s", response.text)
        except requests.exceptions.RequestException as e:
            logger.error("An error occurred while trying to send the request: %s", e)

    def _proxy_setup(self, client_password: str = ""):
        """Setup system-wide proxy configuration using proxy pool
        
        Args:
            client_password (str): Password for sudo operations, defaults to "password"
        """
        retry = 0
        while retry < MAX_RETRIES:
            try:
                _ = requests.get(self.http_server + "/terminal")
                break
            except:
                time.sleep(5)
                retry += 1
                logger.info(f"retry: {retry}/{MAX_RETRIES}")
            
            if retry == MAX_RETRIES:
                return False
            
        # Get proxy from global proxy pool
        proxy_pool = get_global_proxy_pool()
        current_proxy = proxy_pool.get_next_proxy()
        
        if not current_proxy:
            logger.error("No proxy available from proxy pool")
            raise Exception("No proxy available from proxy pool")
        
        # Format proxy URL
        proxy_url = proxy_pool._format_proxy_url(current_proxy)
        logger.info(f"Setting up proxy: {current_proxy.host}:{current_proxy.port}")
        
        # Configure system proxy environment variables  
        proxy_commands = [
            f"echo '{client_password}' | sudo -S bash -c \"apt-get update\"", ## TODO: remove this line if ami is already updated
            f"echo '{client_password}' | sudo -S bash -c \"apt-get install -y tinyproxy\"", ## TODO: remove this line if tinyproxy is already installed
            f"echo '{client_password}' | sudo -S bash -c \"echo 'Port 18888' > /tmp/tinyproxy.conf\"",
            f"echo '{client_password}' | sudo -S bash -c \"echo 'Allow 127.0.0.1' >> /tmp/tinyproxy.conf\"",
            f"echo '{client_password}' | sudo -S bash -c \"echo 'Upstream http {current_proxy.username}:{current_proxy.password}@{current_proxy.host}:{current_proxy.port}' >> /tmp/tinyproxy.conf\"",
            
            # CML commands to set environment variables for proxy
            f"echo 'export http_proxy={proxy_url}' >> ~/.bashrc",
            f"echo 'export https_proxy={proxy_url}' >> ~/.bashrc",
            f"echo 'export HTTP_PROXY={proxy_url}' >> ~/.bashrc",
            f"echo 'export HTTPS_PROXY={proxy_url}' >> ~/.bashrc",
        ]

        # Execute all proxy configuration commands
        for cmd in proxy_commands:
            try:
                self._execute_setup([cmd], shell=True)
            except Exception as e:
                logger.error(f"Failed to execute proxy setup command: {e}")
                proxy_pool.mark_proxy_failed(current_proxy)
                raise
        
        self._launch_setup(["tinyproxy -c /tmp/tinyproxy.conf -d"], shell=True)
        
        # Reload environment variables
        reload_cmd = "source /etc/environment"
        try:
            logger.info(f"Proxy setup completed successfully for {current_proxy.host}:{current_proxy.port}")
            proxy_pool.mark_proxy_success(current_proxy)
        except Exception as e:
            logger.error(f"Failed to reload environment variables: {e}")
            proxy_pool.mark_proxy_failed(current_proxy)
            raise

    # Chrome setup
    def _chrome_open_tabs_setup(self, urls_to_open: List[str]):
        """Open Chrome tabs via in-VM CDP + wmctrl to raise window.

        Playwright CDP from outside the VM fails on AWS because Chrome
        rejects external Host headers with HTTP 400.  Instead we use
        in-VM python3 to execute JavaScript in the existing Chrome tab
        via CDP websocket, which opens new tabs in the same window
        (unlike /json/new which creates orphaned targets).  Then we use
        ``wmctrl`` to bring the Chrome window to focus.
        """
        logger.info("Opening Chrome tabs via in-VM CDP (window.open) + wmctrl …")
        headers = {"Content-Type": "application/json"}

        # Restart Chrome without --proxy-server so pages load via direct
        # internet.  The framework launches Chrome with a proxy flag
        # pointing to tinyproxy → dataimpulse, but if that proxy is
        # broken all pages show "no internet".  Direct internet from
        # AWS works fine for most sites.
        kill_payload = json.dumps({
            "command": "pkill -f 'chrome.*remote-debugging' || true; sleep 2",
            "shell": True,
        })
        requests.post(
            self.http_server + "/setup/execute",
            headers=headers, data=kill_payload, timeout=15,
        )
        relaunch_payload = json.dumps({
            "command": ["google-chrome", "--remote-debugging-port=1337", "--remote-allow-origins=*"],
        })
        requests.post(
            self.http_server + "/setup/launch",
            headers=headers, data=relaunch_payload, timeout=15,
        )
        logger.info("Relaunched Chrome without proxy")

        # Wait for Chrome to be ready inside the VM.
        check_cmd = (
            "python3 -c \""
            "import urllib.request;"
            " r = urllib.request.urlopen('http://localhost:1337/json/version');"
            " print(r.status)"
            "\""
        )
        for attempt in range(15):
            if attempt > 0:
                time.sleep(5)
            check_payload = json.dumps({"command": check_cmd, "shell": True})
            try:
                resp = requests.post(
                    self.http_server + "/setup/execute",
                    headers=headers, data=check_payload, timeout=15,
                )
                if resp.status_code == 200:
                    output = resp.json().get("output", "").strip()
                    if output == "200":
                        break
                    error = resp.json().get("error", "").strip()
                    logger.error(
                        "Attempt %d: Chrome not ready (output=%s, err=%s), retrying …",
                        attempt + 1, repr(output), repr(error[:200]) if error else "",
                    )
                else:
                    logger.error("Attempt %d: execute endpoint returned %s", attempt + 1, resp.status_code)
            except Exception as e:
                logger.error("Attempt %d: in-VM check failed: %s", attempt + 1, e)
        else:
            logger.error("Chrome not reachable after 15 attempts")
            return self._chrome_open_tabs_playwright(urls_to_open)

        # Open tabs using window.open() via CDP websocket.  This creates
        # tabs inside the SAME browser window (unlike /json/new which
        # may create detached page targets).  We also navigate the
        # original blank tab to urls[0] so there's no leftover blank tab.
        #
        # Step 1: Write helper script to the VM.
        import base64 as _b64
        helper_script = r'''
import urllib.request, json, socket, os, struct, base64, sys

urls = json.loads(sys.argv[1])
CDP_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 1337

r = urllib.request.urlopen(f'http://localhost:{CDP_PORT}/json')
tabs = json.loads(r.read())
ws_url = tabs[0]['webSocketDebuggerUrl']

# Parse ws://host:port/path
parts = ws_url.replace('ws://', '').split('/', 1)
hp = parts[0].split(':')
host, port = hp[0], int(hp[1])
path = '/' + parts[1] if len(parts) > 1 else '/'

sock = socket.create_connection((host, port))
key = base64.b64encode(os.urandom(16)).decode()
sock.sendall(f'GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n'.encode())
buf = b''
while b'\r\n\r\n' not in buf:
    buf += sock.recv(4096)

def send(msg):
    d = msg.encode()
    f = bytearray([0x81])
    L = len(d)
    if L < 126: f.append(0x80 | L)
    elif L < 65536: f.append(0x80 | 126); f += struct.pack('>H', L)
    else: f.append(0x80 | 127); f += struct.pack('>Q', L)
    m = os.urandom(4); f += m
    f += bytes(b ^ m[i%4] for i, b in enumerate(d))
    sock.sendall(f)

def recv():
    h = sock.recv(2)
    if len(h) < 2: return ''
    L = h[1] & 0x7f
    if L == 126: L = struct.unpack('>H', sock.recv(2))[0]
    elif L == 127: L = struct.unpack('>Q', sock.recv(8))[0]
    d = b''
    while len(d) < L: d += sock.recv(L - len(d))
    return d.decode()

# Open urls[1:] as new tabs via window.open (attached to same window)
for i, u in enumerate(urls[1:]):
    send(json.dumps({'id': i+1, 'method': 'Runtime.evaluate',
         'params': {'expression': "window.open('" + u + "')"}}))
    recv()

# Navigate the original blank tab to urls[0]
send(json.dumps({'id': 999, 'method': 'Page.navigate',
     'params': {'url': urls[0]}}))
recv()

sock.close()
print(f'Opened {len(urls)} tabs')
'''
        encoded = _b64.b64encode(helper_script.encode()).decode()
        write_payload = json.dumps({
            "command": f"echo '{encoded}' | base64 -d > /tmp/_chrome_open_tabs.py",
            "shell": True,
        })
        requests.post(
            self.http_server + "/setup/execute",
            headers=headers, data=write_payload, timeout=10,
        )

        # Step 2: Execute the helper script.
        urls_json = json.dumps(urls_to_open)
        exec_payload = json.dumps({
            "command": f"python3 /tmp/_chrome_open_tabs.py '{urls_json}' 1337",
            "shell": True,
        })
        opened_via_ws = False
        try:
            resp = requests.post(
                self.http_server + "/setup/execute",
                headers=headers, data=exec_payload, timeout=60,
            )
            if resp.status_code == 200:
                output = resp.json().get("output", "").strip()
                error = resp.json().get("error", "").strip()
                if "Opened" in output:
                    logger.info("Tabs opened via CDP websocket: %s", output)
                    opened_via_ws = True
                else:
                    logger.warning("CDP websocket helper: output=%s err=%s",
                                   repr(output), repr(error[:200]) if error else "")
        except Exception as e:
            logger.warning("CDP websocket helper failed: %s", e)

        # Fallback: /json/new (tabs may be detached — don't close blank tab)
        if not opened_via_ws:
            logger.info("Falling back to /json/new")
            for i, url in enumerate(urls_to_open):
                open_cmd = (
                    f"python3 -c \""
                    f"import urllib.request;"
                    f" urllib.request.urlopen('http://localhost:1337/json/new?{url}');"
                    f" print('OK')"
                    f"\""
                )
                fallback_payload = json.dumps({"command": open_cmd, "shell": True})
                try:
                    resp = requests.post(
                        self.http_server + "/setup/execute",
                        headers=headers, data=fallback_payload, timeout=60,
                    )
                    if resp.status_code == 200:
                        logger.info("Opened tab %d via /json/new: %s", i + 1, url)
                except Exception as e:
                    logger.warning("Failed to open tab %d (%s): %s", i + 1, url, e)

        # Bring Chrome window to focus using wmctrl.
        focus_payload = json.dumps({
            "command": "wmctrl -a Chrome || wmctrl -a chrome || true",
            "shell": True,
        })
        try:
            requests.post(
                self.http_server + "/setup/execute",
                headers=headers, data=focus_payload, timeout=10,
            )
            logger.info("Raised Chrome window via wmctrl")
        except Exception as e:
            logger.warning("Could not raise Chrome window: %s", e)

        # Diagnostic: check Chrome is still running.
        diag_payload = json.dumps({
            "command": "pgrep -c chrome && echo ALIVE || echo DEAD",
            "shell": True,
        })
        try:
            resp = requests.post(
                self.http_server + "/setup/execute",
                headers=headers, data=diag_payload, timeout=10,
            )
            if resp.status_code == 200:
                logger.info("Chrome process check: %s", resp.json().get("output", "").strip())
        except Exception:
            pass

        return None

    def _chrome_open_tabs_playwright(self, urls_to_open: List[str]):
        """Fallback: open Chrome tabs via Playwright CDP (Docker/VMware)."""
        host = self.vm_ip
        port = self.chromium_port
        remote_debugging_url = f"http://{host}:{port}"
        logger.info("Falling back to Playwright CDP @ %s", remote_debugging_url)
        for attempt in range(15):
            if attempt > 0:
                time.sleep(5)

            browser = None
            with sync_playwright() as p:
                try:
                    browser = p.chromium.connect_over_cdp(remote_debugging_url)
                except Exception as e:
                    if attempt < 14:
                        logger.error(f"Attempt {attempt + 1}: Failed to connect, retrying. Error: {e}")
                        continue
                    else:
                        logger.error(f"Failed to connect after multiple attempts: {e}")
                        raise e

                if not browser:
                    return

                logger.info("Opening %s...", urls_to_open)
                for i, url in enumerate(urls_to_open):
                    if i == 0:
                        context = browser.contexts[0]

                    page = context.new_page()
                    try:
                        page.goto(url, timeout=60000)
                    except:
                        logger.warning("Opening %s exceeds time limit", url)
                    logger.info(f"Opened tab {i + 1}: {url}")

                    if i == 0:
                        default_page = context.pages[0]
                        default_page.close()

                return browser, context

    def _chrome_close_tabs_setup(self, urls_to_close: List[str]):
        """Close Chrome tabs matching the given URLs.

        Uses in-VM python3 urllib to localhost CDP first (AWS), falls
        back to Playwright CDP (Docker/VMware).
        """
        time.sleep(5)  # Wait for Chrome to finish launching

        host = self.vm_ip
        port = self.chromium_port
        headers = {"Content-Type": "application/json"}

        # Try in-VM python3 approach first.
        # Build a script that lists tabs, matches URLs, and closes them.
        urls_repr = repr(urls_to_close)
        close_script = (
            "python3 -c \""
            "import urllib.request, json, sys;"
            " r = urllib.request.urlopen('http://localhost:1337/json');"
            " tabs = json.loads(r.read());"
            f" targets = {urls_repr};"
            " closed = [];"
            " [("
            "   urllib.request.urlopen('http://localhost:1337/json/close/'+t['id']),"
            "   closed.append(t['url'])"
            " ) for u in targets for t in tabs"
            "   if u.rstrip('/') in t.get('url','').rstrip('/')];"
            " print(','.join(closed))"
            "\""
        )
        close_payload = json.dumps({"command": close_script, "shell": True})
        try:
            resp = requests.post(
                self.http_server + "/setup/execute",
                headers=headers, data=close_payload, timeout=15,
            )
            if resp.status_code == 200:
                closed = resp.json().get("output", "").strip()
                if closed:
                    logger.info("Closed tabs via in-VM python3: %s", closed)
                return
        except Exception as e:
            logger.warning("In-VM python3 close_tabs failed, falling back to Playwright: %s", e)

        # Fallback: Playwright CDP
        remote_debugging_url = f"http://{host}:{port}"
        with sync_playwright() as p:
            browser = None
            for attempt in range(15):
                try:
                    browser = p.chromium.connect_over_cdp(remote_debugging_url)
                    break
                except Exception as e:
                    if attempt < 14:
                        logger.error(f"Attempt {attempt + 1}: Failed to connect, retrying. Error: {e}")
                        time.sleep(5)
                    else:
                        logger.error(f"Failed to connect after multiple attempts: {e}")
                        raise e

            if not browser:
                return

            for i, url in enumerate(urls_to_close):
                if i == 0:
                    context = browser.contexts[0]

                for page in context.pages:
                    if compare_urls(page.url, url):
                        context.pages.pop(context.pages.index(page))
                        page.close()
                        logger.info(f"Closed tab {i + 1}: {url}")
                        break

            return browser, context

    # google drive setup
    def _googledrive_setup(self, **config):
        """ Clean google drive space (eliminate the impact of previous experiments to reset the environment)
        @args:
            config(Dict[str, Any]): contain keys
                settings_file(str): path to google drive settings file, which will be loaded by pydrive.auth.GoogleAuth()
                operation(List[str]): each operation is chosen from ['delete', 'upload']
                args(List[Dict[str, Any]]): parameters for each operation
            different args dict for different operations:
                for delete:
                    query(str): query pattern string to search files or folder in google drive to delete, please refer to
                        https://developers.google.com/drive/api/guides/search-files?hl=en about how to write query string.
                    trash(bool): whether to delete files permanently or move to trash. By default, trash=false, completely delete it.
                for mkdirs:
                    path(List[str]): the path in the google drive to create folder
                for upload:
                    path(str): remote url to download file
                    dest(List[str]): the path in the google drive to store the downloaded file
        """
        settings_file = config.get('settings_file', 'evaluation_examples/settings/googledrive/settings.yml')
        gauth = GoogleAuth(settings_file=settings_file)
        drive = GoogleDrive(gauth)

        def mkdir_in_googledrive(paths: List[str]):
            paths = [paths] if type(paths) != list else paths
            parent_id = 'root'
            for p in paths:
                q = f'"{parent_id}" in parents and title = "{p}" and mimeType = "application/vnd.google-apps.folder" and trashed = false'
                folder = drive.ListFile({'q': q}).GetList()
                if len(folder) == 0:  # not exists, create it
                    parents = {} if parent_id == 'root' else {'parents': [{'id': parent_id}]}
                    file = drive.CreateFile({'title': p, 'mimeType': 'application/vnd.google-apps.folder', **parents})
                    file.Upload()
                    parent_id = file['id']
                else:
                    parent_id = folder[0]['id']
            return parent_id

        for oid, operation in enumerate(config['operation']):
            if operation == 'delete':  # delete a specific file
                # query pattern string, by default, remove all files/folders not in the trash to the trash
                params = config['args'][oid]
                q = params.get('query', '')
                trash = params.get('trash', False)
                q_file = f"( {q} ) and mimeType != 'application/vnd.google-apps.folder'" if q.strip() else "mimeType != 'application/vnd.google-apps.folder'"
                filelist: GoogleDriveFileList = drive.ListFile({'q': q_file}).GetList()
                q_folder = f"( {q} ) and mimeType = 'application/vnd.google-apps.folder'" if q.strip() else "mimeType = 'application/vnd.google-apps.folder'"
                folderlist: GoogleDriveFileList = drive.ListFile({'q': q_folder}).GetList()
                for file in filelist:  # first delete file, then folder
                    file: GoogleDriveFile
                    if trash:
                        file.Trash()
                    else:
                        file.Delete()
                for folder in folderlist:
                    folder: GoogleDriveFile
                    # note that, if a folder is trashed/deleted, all files and folders in it will be trashed/deleted
                    if trash:
                        folder.Trash()
                    else:
                        folder.Delete()
            elif operation == 'mkdirs':
                params = config['args'][oid]
                mkdir_in_googledrive(params['path'])
            elif operation == 'upload':
                params = config['args'][oid]
                url = params['url']
                with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmpf:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            tmpf.write(chunk)
                    tmpf.close()
                    paths = [params['path']] if params['path'] != list else params['path']
                    parent_id = mkdir_in_googledrive(paths[:-1])
                    parents = {} if parent_id == 'root' else {'parents': [{'id': parent_id}]}
                    file = drive.CreateFile({'title': paths[-1], **parents})
                    file.SetContentFile(tmpf.name)
                    file.Upload()
                return
            else:
                raise ValueError('[ERROR]: not implemented clean type!')

    def _login_setup(self, **config):
        """ Login to a website with account and password information.
        @args:
            config(Dict[str, Any]): contain keys
                settings_file(str): path to the settings file
                platform(str): platform to login, implemented platforms include:
                    googledrive: https://drive.google.com/drive/my-drive

        """
        host = self.vm_ip
        port = self.chromium_port

        remote_debugging_url = f"http://{host}:{port}"
        with sync_playwright() as p:
            browser = None
            for attempt in range(15):
                try:
                    browser = p.chromium.connect_over_cdp(remote_debugging_url)
                    break
                except Exception as e:
                    if attempt < 14:
                        logger.error(f"Attempt {attempt + 1}: Failed to connect, retrying. Error: {e}")
                        time.sleep(5)
                    else:
                        logger.error(f"Failed to connect after multiple attempts: {e}")
                        raise e
            if not browser:
                return

            context = browser.contexts[0]
            platform = config['platform']

            if platform == 'googledrive':
                url = 'https://drive.google.com/drive/my-drive'
                page = context.new_page()  # Create a new page (tab) within the existing context
                try:
                    page.goto(url, timeout=60000)
                except:
                    logger.warning("Opening %s exceeds time limit", url)  # only for human test
                logger.info(f"Opened new page: {url}")
                settings = json.load(open(config['settings_file']))
                email, password = settings['email'], settings['password']

                try:
                    page.wait_for_selector('input[type="email"]', state="visible", timeout=3000)
                    page.fill('input[type="email"]', email)
                    page.click('#identifierNext > div > button')
                    page.wait_for_selector('input[type="password"]', state="visible", timeout=5000)
                    page.fill('input[type="password"]', password)
                    page.click('#passwordNext > div > button')
                    page.wait_for_load_state('load', timeout=5000)
                except TimeoutError:
                    logger.info('[ERROR]: timeout when waiting for google drive login page to load!')
                    return

            else:
                raise NotImplementedError

            return browser, context

    def _update_browse_history_setup(self, **config):
        cache_path = os.path.join(self.cache_dir, "history_new.sqlite")
        db_url = "https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/chrome/44ee5668-ecd5-4366-a6ce-c1c9b8d4e938/history_empty.sqlite?download=true"
        if not os.path.exists(cache_path):
            max_retries = 3
            downloaded = False
            e = None
            for i in range(max_retries):
                try:
                    response = requests.get(db_url, stream=True)
                    response.raise_for_status()

                    with open(cache_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    logger.info("File downloaded successfully")
                    downloaded = True
                    break

                except requests.RequestException as e:
                    logger.error(
                        f"Failed to download {db_url} caused by {e}. Retrying... ({max_retries - i - 1} attempts left)")
            if not downloaded:
                raise requests.RequestException(f"Failed to download {db_url}. No retries left. Error: {e}")
        else:
            logger.info("File already exists in cache directory")
        # copy a new history file in the tmp folder
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "history_empty.sqlite")
            shutil.copy(cache_path, db_path)

            history = config['history']

            for history_item in history:
                url = history_item['url']
                title = history_item['title']
                visit_time = datetime.now() - timedelta(seconds=history_item['visit_time_from_now_in_seconds'])

                # Chrome use ms from 1601-01-01 as timestamp
                epoch_start = datetime(1601, 1, 1)
                chrome_timestamp = int((visit_time - epoch_start).total_seconds() * 1000000)

                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO urls (url, title, visit_count, typed_count, last_visit_time, hidden)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (url, title, 1, 0, chrome_timestamp, 0))

                url_id = cursor.lastrowid

                cursor.execute('''
                    INSERT INTO visits (url, visit_time, from_visit, transition, segment_id, visit_duration)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (url_id, chrome_timestamp, 0, 805306368, 0, 0))

                conn.commit()
                conn.close()

            logger.info('Fake browsing history added successfully.')

            controller = PythonController(self.vm_ip, self.server_port)

            # get the path of the history file according to the platform
            os_type = controller.get_vm_platform()

            if os_type == 'Windows':
                chrome_history_path = controller.execute_python_command(
                    """import os; print(os.path.join(os.getenv('USERPROFILE'), "AppData", "Local", "Google", "Chrome", "User Data", "Default", "History"))""")[
                    'output'].strip()
            elif os_type == 'Darwin':
                chrome_history_path = controller.execute_python_command(
                    """import os; print(os.path.join(os.getenv('HOME'), "Library", "Application Support", "Google", "Chrome", "Default", "History"))""")[
                    'output'].strip()
            elif os_type == 'Linux':
                arch = controller.get_vm_machine().lower()
                if 'arm' in arch or 'aarch' in arch:
                    chrome_history_path = controller.execute_python_command(
                        "import os; print(os.path.join(os.getenv('HOME'), 'snap', 'chromium', 'common', 'chromium', 'Default', 'History'))")[
                        'output'].strip()
                else:
                    chrome_history_path = controller.execute_python_command(
                        "import os; print(os.path.join(os.getenv('HOME'), '.config', 'google-chrome', 'Default', 'History'))")[
                        'output'].strip()
            else:
                raise Exception('Unsupported operating system')

            form = MultipartEncoder({
                "file_path": chrome_history_path,
                "file_data": (os.path.basename(chrome_history_path), open(db_path, "rb"))
            })
            headers = {"Content-Type": form.content_type}
            logger.debug(form.content_type)

            # send request to server to upload file
            try:
                logger.debug("REQUEST ADDRESS: %s", self.http_server + "/setup" + "/upload")
                response = requests.post(self.http_server + "/setup" + "/upload", headers=headers, data=form)
                if response.status_code == 200:
                    logger.info("Command executed successfully: %s", response.text)
                else:
                    logger.error("Failed to upload file. Status code: %s", response.text)
            except requests.exceptions.RequestException as e:
                logger.error("An error occurred while trying to send the request: %s", e)

            self._execute_setup(["sudo chown -R user:user /home/user/.config/google-chrome/Default/History"], shell=True)

    def _google_sheet_from_template_setup(self, **config):
        """No-op setup function for google_sheet_from_template.

        The actual sheet creation is handled in run_parallel_task.py and run_baseline_task.py
        via _process_google_workspace_config before env.reset() is called.
        """
        pass

    def _google_doc_from_template_setup(self, **config):
        """No-op setup function for google_doc_from_template.

        The actual doc creation is handled in run_parallel_task.py and run_baseline_task.py
        via _process_google_workspace_config before env.reset() is called.
        """
        pass
