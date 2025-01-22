FROM ubuntu:16.04
LABEL maintainer="Swathi Thiruvengadam <thiruves@tf.uni-freiburg.de>"
RUN apt-get update && apt-get install -y make vim && rm -rf /var/lib/apt/lists/*
COPY Makefile Makefile
COPY bashrc bashrc
CMD ["/bin/bash", "--rcfile", "bashrc"]

docker build -t Swathi-Thiruvengadam-[project] .
# docker run -it -v $(pwd)S:\23502\2\280_PVM\Aktuell\01_Orga\23131_MAS_TeamModulbewertung\03_Arbeitsordner\Swathi_Thiruvengadam\table-transformer:\table-transformer:ro  --name Swathi-Thiruvengadam-[project] 