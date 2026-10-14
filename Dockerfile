# Flood damage assessment: API + dashboard + pipeline CLI in one image.
#   docker build -t trishuli-flood .
# Base image pinned to the 1.5 line of micromamba (the plan's FROM mambaorg/micromamba:1.5), exact patch release for reproducibility.
FROM mambaorg/micromamba:1.5.10

# Devanagari fonts for the Nepali PDF (WeasyPrint uses system fonts; pango itself comes from conda-forge).
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-noto-core fonts-noto-ui-core \
 && rm -rf /var/lib/apt/lists/*

# OPTIONAL, UNTESTED: headless ESA SNAP for Sentinel-1 preprocessing (the `s1` pipeline stage needs `gpt`).
# Pass the URL of the SNAP "Linux Unix" installer you downloaded from https://step.esa.int:
#   docker build --build-arg SNAP_INSTALLER_URL=<url> -t trishuli-flood .
# Without it the image still runs the API, dashboard, demo and all post-preprocessing stages; mount your own SNAP otherwise.
ARG SNAP_INSTALLER_URL=""
RUN if [ -n "$SNAP_INSTALLER_URL" ]; then \
      apt-get update && apt-get install -y --no-install-recommends curl default-jre-headless \
      && curl -fsSL "$SNAP_INSTALLER_URL" -o /tmp/snap.sh \
      && sh /tmp/snap.sh -q -dir /opt/snap \
      && rm -f /tmp/snap.sh && rm -rf /var/lib/apt/lists/* \
      && ln -s /opt/snap/bin/gpt /usr/local/bin/gpt ; \
    fi
USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean -a -y

COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
WORKDIR /app
ENV FLOOD_OUT=data/outputs FLOOD_CONFIG=config/aoi_trishuli.yaml
EXPOSE 8000 8501
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
