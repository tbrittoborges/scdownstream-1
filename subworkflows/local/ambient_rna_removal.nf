include { CELDA_DECONTX               } from '../../modules/local/celda/decontx'
include { CELLBENDER_REMOVEBACKGROUND } from '../../modules/local/cellbender/removebackground'
include { CELLBENDER_MERGE            } from '../../modules/local/cellbender/merge'

workflow AMBIENT_RNA_REMOVAL {
    take:
    ch_h5ad

    main:
    ch_versions = Channel.empty()

    if (params.ambient_removal == 'decontx') {
        CELDA_DECONTX(ch_h5ad)
        ch_h5ad = CELDA_DECONTX.out.h5ad
        ch_versions = ch_versions.mix(CELDA_DECONTX.out.versions)
    }

    if (params.ambient_removal == 'cellbender') {
        CELLBENDER_REMOVEBACKGROUND(ch_h5ad)
        ch_versions = ch_versions.mix(CELLBENDER_REMOVEBACKGROUND.out.versions)

        CELLBENDER_MERGE(ch_h5ad.join(CELLBENDER_REMOVEBACKGROUND.out.h5))
        ch_h5ad = CELLBENDER_MERGE.out.h5ad
        ch_versions = ch_versions.mix(CELLBENDER_MERGE.out.versions)
    }

    emit:
    h5ad = ch_h5ad

    versions = ch_versions
}
