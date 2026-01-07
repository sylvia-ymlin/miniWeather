
#!/bin/bash
set -e

echo "🚀 Building and starting MPI Cluster..."
docker-compose up -d --build

echo "⏳ Waiting for nodes to initialize..."
sleep 5

echo "🔑 Configuring SSH Trust between nodes..."
# Copy master's public key to worker so master can ssh to worker
docker cp mpi-master:/root/.ssh/id_rsa.pub ./master_key.pub
docker cp ./master_key.pub mpi-worker:/root/.ssh/authorized_keys
rm ./master_key.pub

# Fix permissions
docker exec mpi-worker chmod 600 /root/.ssh/authorized_keys
docker exec mpi-worker chown root:root /root/.ssh/authorized_keys

echo "✅ Cluster is Ready!"
echo ""
echo "To enter the cluster:"
echo "  docker exec -it mpi-master bash"
echo ""
echo "To run MPI test:"
echo "  (Inside master): mpirun -host master,worker -n 4 /app/build/miniWeather_mpi"
