import sys
import xtdata, xtmodel, vtkgeo

# sometimes this seems necessary
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":    
    data = xtdata.XtomicsData(
        array_fname='../allen/xtomics_data/array.h5',
        array_ds='data',
        labels_ds='sample_name',
        metadata_fname='../allen/metadata.csv'
    )

    mc = xtmodel.ModelConfig(
        data=data,
        output_path='./output')

    batch_size = 256
    epochs = 20

    cmd = sys.argv[1]
    if cmd == 'train':
        mc.train(batch_size=batch_size, epochs=epochs)
    elif cmd == 'geo':
        vtkgeo.build_geo(mc, "test_geo.ply", sample_size=100000)




