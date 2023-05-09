g++ -pthread src/orca-server-mahimahi.cc src/flow.cc -o orca-server-mahimahi
g++ src/client.c -o client
cp client rl_module/
mv orca-server*  rl_module/
sudo chmod +x rl_module/client
sudo chmod +x rl_module/orca-server-mahimahi


