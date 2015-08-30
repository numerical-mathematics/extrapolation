fnbod.so: fnbod.f
	f2py -c fnbod.f -m fnbod

clean:
	rm fnbod.so
