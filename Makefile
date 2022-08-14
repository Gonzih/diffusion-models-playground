convert-svgs: cp-svgs images-ready/*.svg

cp-svgs:
	cp images-svg/*.svg images-ready

.PHONY: images-ready/*.svg
images-ready/*.svg:
	convert -define svg:size=400x400 $@ -thumbnail 100x100^ -gravity center -extent 128x128  $(subst .svg,.jpeg,$@)
	rm $@ -f

images.tar.gz:
	tar cvzf $@ images-ready
