Peculiarities from libaries
===========================

Image origin and axis order in different libraries:

    * PIL(LOW):     top left    hor,ver
    * MATLAB:       top left    ver,hor
    * Matplotlib:   top left    ver,hor? (change O with imshow(origin='lower'))

Read and show image:

    * PIL(LOW):     Image.open({file})      {img}.show()
    * Matplotlib:   image.imread({file})    pyplot.imshow({img}); pyplot.show()
    * MATLAB:       imread({file})          imshow({img})
    * Scipy.ndimage imread({file})          -
    * OpenCV:       imread({file})          imshow({wname}, {img})

