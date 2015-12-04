fnbod.so: fnbod.f
	f2py -c fnbod.f -m fnbod

fnbruss.so: fnbruss.f
	f2py -c fnbruss.f -m fnbruss

clean:
	rm fnbod.so
	rm fnbruss.so
