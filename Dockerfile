# Use Python 3.12 slim image
FROM python:3.12-slim

# Install cron and other system dependencies
RUN apt-get update && apt-get install -y \
    cron \
    curl \
    gnupg2 \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft ODBC Driver for SQL Server
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY Clustering_Analysis.py ./
COPY etl.py ./
COPY src/ ./src/
COPY pages/ ./pages/
COPY data/ ./data/

# Install uv package manager
RUN pip install --no-cache-dir uv

# Install Python dependencies
RUN uv pip install --system -r pyproject.toml

# Create cron job file for monthly ETL
RUN echo "0 0 1 * * cd /app && python etl.py >> /var/log/cron.log 2>&1" > /etc/cron.d/etl-cron

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/etl-cron

# Apply cron job
RUN crontab /etc/cron.d/etl-cron

# Create log file
RUN touch /var/log/cron.log

# Expose Streamlit default port
EXPOSE 8501

# Create startup script
RUN echo '#!/bin/bash\n\
# Start cron in the background\n\
cron\n\
\n\
# Start Streamlit\n\
streamlit run Clustering_Analysis.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"]
