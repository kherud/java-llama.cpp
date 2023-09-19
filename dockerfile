FROM maven:3.9.3-eclipse-temurin-17-focal
RUN apt update && apt install -y git build-essential cmake
COPY pom.xml .
COPY .git ./.git
COPY src ./src
RUN git submodule update --init
COPY scripts ./scripts
RUN ./scripts/build-so.sh

