process SCANPY_HVGS {
    tag "$meta.id"
    label 'process_medium'
    label 'process_gpu'

    conda "${moduleDir}/environment.yml"
    container "${ task.ext.use_gpu ? 'docker.io/nicotru/rapids-singlecell:0.0.1' :
        workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'oras://community.wave.seqera.io/library/python-igraph_scanpy:a7114d55b0af3893':
        'community.wave.seqera.io/library/python-igraph_scanpy:5f677450e42211ef' }"

    input:
    tuple val(meta), path(h5ad)
    val(n_hvgs)

    output:
    tuple val(meta), path("*.h5ad"), emit: h5ad
    path "versions.yml"            , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    template 'hvgs.py'
}
