

# refresh your system
$ sudo apt-get update
# need nano for editing some files
$ sudo apt-get install nano
$ sudo apt-get upgrade
$ sudo apt-get autoremove

sudo nano /etc/update-manager/release-upgrades

# refresh your system again
$ sudo apt-get update
$ sudo apt-get dist-upgrade
$ sudo reboot

# upgrade to Ubuntu 20.04 - Don't reboot immediately
$ sudo do-release-upgrade

# check and editing some files
$ sudo nano /etc/gdm3/custom.conf                   # Set WaylandEnable=false
$ sudo nano /etc/X11/xorg.conf                      # Uncomment Driver "nividia"
$ sudo nano /etc/update-manager/release-upgrades    # Reset prompt=never
$ sudo reboot


# prepare your system
$ sudo apt-get update
$ sudo apt-get upgrade
$ sudo apt-get autoremove
# remove circular symlink
$ sudo rm /usr/share/applications/vpi1_demos
# remove distorted nvidia logo in top bar
$ cd /usr/share/nvpmodel_indicator
$ sudo mv nv_logo.svg no_logo.svg



# install gcc and g++ version 8
$ sudo apt-get install gcc-8 g++-8 clang-8
# setup the gcc selector
$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
# setup the g++ selector
$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
# setup the clang selector
$ sudo update-alternatives --install /usr/bin/g++ clang /usr/bin/clang-10 10
$ sudo update-alternatives --install /usr/bin/g++ clang /usr/bin/clang-8 8
# if you want to make a selection use these commands
$ sudo update-alternatives --config gcc
$ sudo update-alternatives --config g++
$ sudo update-alternatives --config clang


sudo apt --fix-broken