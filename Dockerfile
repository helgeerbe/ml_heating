ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Use architecture-specific base images
FROM --platform=$TARGETPLATFORM python:3.11-alpine3.18

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
ENV LANG=C.UTF-8

# Install system dependencies
RUN apk add --no-cache \
    bash \
    curl \
    jq \
    tzdata \
    procps \
    supervisor \
    gcc \
    g++ \
    musl-dev \
    linux-headers \
    gfortran \
    openblas-dev \
    lapack-dev

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt

# Copy the ML heating system source code
COPY src/ /app/src/
COPY notebooks/ /app/notebooks/

# Copy add-on specific files
COPY run.sh /app/
COPY config_adapter.py /app/
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create necessary directories
RUN mkdir -p /data/models \
    && mkdir -p /data/backups \
    && mkdir -p /data/logs \
    && mkdir -p /data/config

# Copy dashboard files
RUN mkdir -p /app/dashboard
COPY dashboard/ /app/dashboard/

# Make run script executable
RUN chmod a+x /app/run.sh

# Health check - use dedicated health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3002/health || exit 1

# Expose health check port only (dashboard now uses ingress)
# Development API (if enabled) will use port 3003
EXPOSE 3002 3003

# Set the entry point
CMD ["/app/run.sh"]
