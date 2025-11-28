ARG BUILD_FROM
FROM $BUILD_FROM

# Install Python dependencies
RUN apk add --no-cache \
    python3 \
    python3-dev \
    py3-pip \
    gcc \
    g++ \
    musl-dev \
    linux-headers

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY notebooks/ /app/notebooks/
COPY run.sh /app/

# Make run script executable
RUN chmod a+x /app/run.sh

# Set up data directory
RUN mkdir -p /data

CMD [ "/app/run.sh" ]
