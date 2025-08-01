from plyfile import PlyData

ply = PlyData.read('convert/broken.ply')
print(ply['vertex'].data.dtype.names)
