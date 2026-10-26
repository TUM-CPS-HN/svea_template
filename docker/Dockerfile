# Dockerfile for SVEA images
#
# Author: Kaj Munhoz Arfvidsson

ARG BUILD_TAG
ARG ROSDISTRO

FROM ${BUILD_TAG:-ghcr.io/kth-sml/svea:latest}

ARG ROSDISTRO
ARG DEBIAN_FRONTEND=noninteractive

## Update Infrastructure (may have changed from base build)

COPY entrypoint ./

## Setup Development Environment

RUN apt-get update -y

# Pin MAVROS to 2.14 (from the ROS 2026-06-18 snapshot).
# MAVROS 2.16 changed the topic namespaces, which breaks lli.xml.
RUN echo "deb [trusted=yes] http://snapshots.ros.org/$ROSDISTRO/2026-06-18/ubuntu noble main" \
        > /etc/apt/sources.list.d/ros-snapshot.list && \
    printf 'Package: ros-%s-mavros* ros-%s-libmavconn ros-%s-mavlink\nPin: origin snapshots.ros.org\nPin-Priority: 1001\n' \
        $ROSDISTRO $ROSDISTRO $ROSDISTRO > /etc/apt/preferences.d/ros-snapshot-mavros && \
    apt-get update -y && \
    apt-get install --no-install-recommends -y --allow-downgrades \
        ros-$ROSDISTRO-mavros ros-$ROSDISTRO-mavros-extras ros-$ROSDISTRO-mavros-msgs \
        ros-$ROSDISTRO-libmavconn ros-$ROSDISTRO-mavlink && \
    apt-mark hold \
        ros-$ROSDISTRO-mavros ros-$ROSDISTRO-mavros-extras ros-$ROSDISTRO-mavros-msgs \
        ros-$ROSDISTRO-libmavconn ros-$ROSDISTRO-mavlink

# # You can add extra apt dependencies like this...
# RUN apt-get install --no-install-recommends -y \
#         tmux htop

# Install pip packages specified in requirements.txt
RUN --mount=type=bind,source=./requirements.txt,target=/tmp/requirements.txt,readonly \
    pip install \
        --break-system-packages \
        -r /tmp/requirements.txt

# (Pre-)Build the ROS 2 workspace 
# This improves build speeds later on.
RUN --mount=type=bind,source=./src,target=./src,readonly /bin/bash <<EOF
set -e
source /opt/svea/$ROSDISTRO/setup.bash
rosdep install \
    --rosdistro $ROSDISTRO \
    --from-paths src \
    --ignore-src \
    -qry
colcon build --symlink-install
EOF

