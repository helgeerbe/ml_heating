FROM python:3.11-alpine3.18

# Build arguments
ARG BUILD_ARCH
ARG BUILD_DATE
ARG BUILD_DESCRIPTION
ARG BUILD_NAME
ARG BUILD_REF
ARG BUILD_REPOSITORY
ARG BUILD_VERSION

# Labels
LABEL \
    io.hass.name="${BUILD_NAME}" \
    io.hass.description="${BUILD_DESCRIPTION}" \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.type="addon" \
    io.hass.version=${BUILD_VERSION} \
    maintainer="ML Heating Contributors" \
    org.opencontainers.image.title="${BUILD_NAME}" \
    org.opencontainers.image.description="${BUILD_DESCRIPTION}" \
    org.opencontainers.image.vendor="Home Assistant Community Add-ons" \
    org.opencontainers.image.authors="ML Heating Contributors" \
    org.opencontainers.image.licenses="MIT" \
    org.opencontainers.image.url="https://github.com/helgeerbe/ml_heating" \
    org.opencontainers.image.source="https://github.com/helgeerbe/ml_heating" \
    org.opencontainers.image.documentation="https://github.com/helgeerbe/ml_heating/blob/main/README.md" \
    org.opencontainers.image.created=${BUILD_DATE} \
    org.opencontainers.image.revision=${BUILD_REF} \
    org.opencontainers.image.version=${BUILD_VERSION}

# Environment
ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1

# Install system dependencies for ML workload and HA addon support
RUN apk add --no-cache \
    bash \
    curl \
    jq \
    tzdata \
    gcc \
    g++ \
    musl-dev \
    linux-headers \
    gfortran \
    openblas-dev \
    lapack-dev \
    && rm -rf /var/cache/apk/*

# Install bashio for Home Assistant addon support
RUN curl -L -s -o /tmp/bashio.tar.gz \
    "https://github.com/hassio-addons/bashio/archive/v0.16.2.tar.gz" \
    && mkdir /tmp/bashio \
    && tar zxf /tmp/bashio.tar.gz -C /tmp/bashio --strip-components 1 \
    && mv /tmp/bashio/lib /usr/lib/bashio \
    && ln -s /usr/lib/bashio/bashio /usr/bin/bashio \
    && rm -rf /tmp/bashio.tar.gz /tmp/bashio

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Copy the ML heating system source code
COPY src/ /app/src/
COPY notebooks/ /app/notebooks/
COPY dashboard/ /app/dashboard/

# Copy configuration adapter and utilities
COPY config_adapter.py /app/
COPY validate_container.py /app/

# Copy and setup entrypoint
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

# Create necessary directories
RUN mkdir -p /data/models \
    && mkdir -p /data/backups \
    && mkdir -p /data/logs \
    && mkdir -p /data/config

# Health check - use dedicated health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3002/health || exit 1

# Expose ports (health check and optional dev API)
EXPOSE 3002 3003

# Use simple entrypoint
CMD ["/app/run.sh"]
