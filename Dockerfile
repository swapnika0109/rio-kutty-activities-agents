# Use python slim for smaller image size (Cost/CO2 optimization)
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies using uv
# --frozen ensures we stick to the lockfile versions
# --no-dev excludes dev dependencies
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ /app/src/

# Expose port
EXPOSE 8080

# Run the application using uv run
# We use uvicorn explicitly via uv run for the production server
# Ensure PATH includes the virtualenv created by uv
ENV PATH="/app/.venv/bin:$PATH"

CMD ["sh", "-c", "uv run uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8080}"]