# Interface for exporting Carla simulator data in DGP and SynchronizedScene formats

**This is a work in progress. There may be many bugs. This is not tested. Use at your own risk!**

This requires carla simulator >= 0.9.13. For a quick setup build the included dockerfile (this pulls carla simulator and adds addtional maps).

```bash
make docker-build-carla
```

Then install python carla api from pip:

```bash
pip3 install carla
```

If using the included docker you can start the Carla simulator on the default port 2000 via
```bash
make docker-run-carla
```

For examples of creating an agent and exporting some data, please see the notebooks section. You may need to forward the carla port so the notebook can connect

```bash
ssh -L 2000:localhost:2000 -L 2001:localhost:2001 -L 2002:localhost:2002 <user>@<whereever-carla-is>
```

Carla (https://carla.org/) is awesome, lots of thanks to the Carla team for all the hard work.