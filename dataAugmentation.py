from   PIL     import  Image
import glob

for imgName in glob.glob('*.png'): 
    img = Image.open(imgName) 
    img.rotate( 5  ).save( imgName[:-4] +  "-rot1"  + ".png" )
    img.rotate( 10 ).save( imgName[:-4] +  "-rot2"  + ".png" )
    img.rotate( 15 ).save( imgName[:-4] +  "-rot3"  + ".png" )
    img.rotate( 20 ).save( imgName[:-4] +  "-rot4"  + ".png" )
    img.rotate( 25 ).save( imgName[:-4] +  "-rot5"  + ".png" )
    img.rotate( 30 ).save( imgName[:-4] +  "-rot6"  + ".png" )
    img.rotate( 35 ).save( imgName[:-4] +  "-rot7"  + ".png" )
    img.rotate( 40 ).save( imgName[:-4] +  "-rot8"  + ".png" )
    img.rotate( 45 ).save( imgName[:-4] +  "-rot9"  + ".png" )
    
    img.rotate( -5  ).save( imgName[:-4] + "-rot10" + ".png" )
    img.rotate( -10 ).save( imgName[:-4] + "-rot11" + ".png" )
    img.rotate( -15 ).save( imgName[:-4] + "-rot12" + ".png" )
    img.rotate( -20 ).save( imgName[:-4] + "-rot13" + ".png" )
    img.rotate( -25 ).save( imgName[:-4] + "-rot14" + ".png" )
    img.rotate( -30 ).save( imgName[:-4] + "-rot15" + ".png" )
    img.rotate( -35 ).save( imgName[:-4] + "-rot16" + ".png" )
    img.rotate( -40 ).save( imgName[:-4] + "-rot17" + ".png" )
    img.rotate( -45 ).save( imgName[:-4] + "-rot18" + ".png" )
    