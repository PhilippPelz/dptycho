package = "dptycho"
version = "scm-1"

source = {
   url = "git://github.com/soumith/examplepackage.torch",
   tag = "master"
}

description = {
   summary = "Deep ptychography package",
   detailed = [[
   	    Deep electron microscopy toolbox.
   ]],
   homepage = "https://google.com"
}

dependencies = {
   "torch >= 7.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
