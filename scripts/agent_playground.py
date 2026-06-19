import os, shutil, subprocess
from pathlib import Path
import requests

import html_to_markdown
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

converter = PdfConverter(
    artifact_dict=create_model_dict(),
)
from io import BytesIO

from dotenv import load_dotenv

load_dotenv()

DEFAULT_PLAYGROUND_PATH = os.path.join(
    os.path.dirname(__file__), "..", ".data", "agent-playground"
)
DOCKERFILE_PATH = os.path.join(os.path.dirname(__file__), "agent_playground.Dockerfile")
DOCKER_CONTAINER_TAG = "yades:agent-playground"

# To avoid getting blocked by websites, we use a user agent string.
USER_AGENT = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "priority": "u=0, i",
    "sec-ch-ua": '"Google Chrome";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
    "sec-ch-ua-arch": '"arm"',
    "sec-ch-ua-bitness": '"64"',
    "sec-ch-ua-full-version-list": '"Google Chrome";v="141.0.7390.123", "Not?A_Brand";v="8.0.0.0", "Chromium";v="141.0.7390.123"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-model": '""',
    "sec-ch-ua-platform": '"macOS"',
    "sec-ch-ua-platform-version": '"15.6.1"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
}


class AgentPlayground:
    """
    Isolates a working director for agent use (download files from the internet, run arbitrary commands, and read files)
    Requires docker to be running. By default uses an image with python3.9, git, and bash pre-installed.
    """

    def __init__(
        self,
        playground_path=DEFAULT_PLAYGROUND_PATH,
        base_image="python:3.9-slim-bullseye",
        install="RUN apt-get update && apt-get install -y --no-install-recommends git bash && rm -rf /var/lib/apt/lists/*",
        hide_build_output=True,
    ):
        """Instantiates playground_path as the working directory for the agent"""

        self.playground_path = playground_path

        os.makedirs(self.playground_path, exist_ok=True)

        with open(DOCKERFILE_PATH, "w") as f:
            f.write(f"FROM {base_image}\n{install}\nWORKDIR /app")

        subprocess.run(
            [
                "docker",
                "build",
                "-f",
                DOCKERFILE_PATH,
                ".",
                "-t",
                DOCKER_CONTAINER_TAG,
            ],
            check=True,
            capture_output=hide_build_output,
        )

    def clean(self):
        """Clears the content of playground_path"""

        shutil.rmtree(self.playground_path)
        os.makedirs(self.playground_path)

    def _resolve_relative_path(self, relative_path: str):
        path = os.path.join(self.playground_path, relative_path)
        assert Path(path).is_relative_to(
            self.playground_path
        ), "must not access file outside playground"
        return path

    def open(self, relative_path: str, mode: str):
        """Returns an open file handle for any path inside playground_path"""

        return open(self._resolve_relative_path(relative_path), mode)

    def run(self, cmd: list[str], timeout_seconds=120):
        """Runs an arbitrary command with playground_path as the working directory."""

        res = subprocess.run(
            [
                "docker",
                "run",
                "--mount",
                f"type=bind,source={self.playground_path},target=/app",
                DOCKER_CONTAINER_TAG,
                *cmd,
            ],
            capture_output=True,
            timeout=timeout_seconds,
        )
        return res.stdout.decode(), res.stderr.decode(), res.returncode

    def download_url_as_markdown(self, url, destination_path, timeout_seconds=120):
        """Scrapes a website url and downloads it as markdown"""

        destination_path = self._resolve_relative_path(destination_path)

        try:
            response = requests.get(url, timeout=timeout_seconds, headers=USER_AGENT)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            text = ""
            if ".pdf" in url:
                rendered = converter(BytesIO(response.content))
                text, _, images = text_from_rendered(rendered)
            else:
                text = html_to_markdown.convert(
                    response.content.decode("utf-8", errors="ignore")
                )

            with open(destination_path, "w") as f:
                f.write(text)

            return True
        except Exception as e:
            print(e)
            return False

    def search_google(self, query: str, num_results: int = 5) -> str:
        """Use the Google search engine to find urls matching a query (with a title and relevant snippet)"""

        search_url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?q={query}"
            f"&key={os.environ['GOOGLE_SEARCH_API_KEY']}"
            f"&cx={os.environ['GOOGLE_SEARCH_ENGINE_ID']}"
            f"&num={num_results}"
        )
        response = requests.get(search_url)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("items", []):
            results.append(
                f"* {item.get('title')} ({item.get('link')}): {item.get('snippet')}"
            )
        return "\n".join(results) if results else "No relevant results found."

    def copy_to_playground(self, src_path: str, dest_path: str):
        """Copies the given src_path to the dest_path. src_path can be a directory. The parent directory of dest_path must already exist"""

        dest_path = self._resolve_relative_path(dest_path)
        if os.path.isdir(src_path):
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            shutil.copytree(src_path, dest_path)
        else:
            shutil.copy(src_path, dest_path)


if __name__ == "__main__":
    # Example usage
    playground = AgentPlayground()
    playground.clean()

    with playground.open("test.py", "w") as f:
        f.write("print('hello world!')")

    out, err, returncode = playground.run(["python", "test.py"])
    print("OUTPUT:")
    print(out)
    if returncode != 0:
        print(f"ERROR (exitcode={returncode}):")
        print(err)

    # playground.download_url_as_markdown("https://koellabs.com", "koellabs.md")

    # print(playground.search_google("MAX78000 datasheet"))
    # playground.download_url_as_markdown(
    #     "https://www.analog.com/media/en/technical-documentation/data-sheets/MAX78000.pdf",
    #     "MAX78000_datasheet.md",
    # )
