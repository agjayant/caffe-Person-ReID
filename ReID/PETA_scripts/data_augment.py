import numpy as np
import Image
import ImageOps

#horizontal mirroring
def mymirror(filename,outname):
    img= Image.open(filename)
    mirror_img = ImageOps.mirror(img)
    mir_img = np.array(mirror_img)
    m_img = Image.fromarray(mir_img , mode='RGB')
    m_img.save(outname)

#translation by (w,h) for each (x,y)
def translate(filename,outname, w, h):
    img= Image.open(filename)
    img = img.transform(img.size, Image.AFFINE, (1,0,w,0,1,h))
    img.save(outname)

