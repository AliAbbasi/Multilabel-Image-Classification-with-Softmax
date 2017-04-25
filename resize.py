from  PIL  import  Image
import  glob


size  = 32,32

for filename in glob.glob('*.png'):
    img = Image.open(filename)
    img.thumbnail(size, Image.ANTIALIAS)
    img.save( filename  )
