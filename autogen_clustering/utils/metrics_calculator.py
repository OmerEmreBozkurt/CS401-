import os
import subprocess
import tempfile
import shutil

class MetricsCalculator:
    """Calculates TurboMQ and MoJo-FM metrics"""
    
    def __init__(self, dependency_rsf_path, experiments_dir=None):
        self.dependency_rsf_path = os.path.abspath(dependency_rsf_path) if dependency_rsf_path else None
        self.experiments_dir = experiments_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "experiments"
        )
        
        # Validate dependency RSF exists
        if self.dependency_rsf_path and not os.path.exists(self.dependency_rsf_path):
            pass
            # print(f"⚠ Warning: Dependency RSF not found: {self.dependency_rsf_path}")
            # print(f"   Current working directory: {os.getcwd()}")
            # Try to suggest correct path
            # dirname = os.path.dirname(self.dependency_rsf_path)
            # if os.path.exists(dirname):
            #     print(f"   Available files in {dirname}:")
            #     for f in os.listdir(dirname):
            #         if f.endswith('.rsf'):
            #             print(f"     - {f}")
    
    def clusters_to_rsf(self, clusters, output_path):
        """Convert clusters dict to RSF format"""
        with open(output_path, 'w') as f:
            for cluster_name, nodes in clusters.items():
                for node in nodes:
                    f.write(f"contain {cluster_name} {node}\n")
        # print(f"✓ Converted clusters to RSF: {output_path}")
    
    def calculate_turbomq(self, clustering_rsf_path):
        """Calculate TurboMQ score using Java jar"""
        turbomq_jar = os.path.join(self.experiments_dir, "turbomq.jar")
        
        if not os.path.exists(turbomq_jar):
            # print(f"⚠ Warning: {turbomq_jar} not found, skipping TurboMQ")
            return None
        
        # Validate input files exist
        if not os.path.exists(self.dependency_rsf_path):
            # print(f"⚠ TurboMQ error: Dependency RSF not found: {self.dependency_rsf_path}")
            return None
        
        if not os.path.exists(clustering_rsf_path):
            # print(f"⚠ TurboMQ error: Clustering RSF not found: {clustering_rsf_path}")
            return None
        
        try:
            # Copy files to experiments directory to avoid path issues with the jar
            dep_rsf_temp = os.path.join(self.experiments_dir, "temp_dependency.rsf")
            clust_rsf_temp = os.path.join(self.experiments_dir, "temp_clustering.rsf")
            
            shutil.copy2(self.dependency_rsf_path, dep_rsf_temp)
            shutil.copy2(clustering_rsf_path, clust_rsf_temp)
            
            # Debug output
            cmd = ["java", "-jar", "turbomq.jar", "temp_dependency.rsf", "temp_clustering.rsf"]
            print(f"   Executing: {' '.join(cmd)}")
            print(f"   CWD: {self.experiments_dir}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.experiments_dir
            )
            
            # Clean up temp files
            try:
                if os.path.exists(dep_rsf_temp):
                    os.remove(dep_rsf_temp)
                if os.path.exists(clust_rsf_temp):
                    os.remove(clust_rsf_temp)
            except:
                pass
            
            # Debug output
            if result.stderr:
                print(f"   TurboMQ stderr: {result.stderr[:200]}")
            
            if result.returncode == 0:
                try:
                    score = float(result.stdout.strip())
                    return score
                except ValueError as e:
                    print(f"⚠ TurboMQ error: Could not parse output as float: {result.stdout.strip()}")
                    return None
            else:
                print(f"⚠ TurboMQ error (exit code {result.returncode}): {result.stderr}")
                return None
        except Exception as e:
            print(f"⚠ TurboMQ calculation failed: {e}")
            return None
    
    def calculate_mojo_fm(self, clustering_rsf_path, reference_rsf_path=None):
        """Calculate MoJo-FM distance using Java jar"""
        mojo_jar = os.path.join(self.experiments_dir, "mojo.jar")
        
        if not os.path.exists(mojo_jar):
            # print(f"⚠ Warning: {mojo_jar} not found, skipping MoJo-FM")
            return None
        
        # If no reference, skip MoJo-FM (it needs a reference clustering)
        if not reference_rsf_path or not os.path.exists(reference_rsf_path):
            # print(f"⚠ No reference RSF provided, skipping MoJo-FM")
            return None
        
        try:
            # Use absolute paths
            clust_rsf_abs = os.path.abspath(clustering_rsf_path)
            ref_rsf_abs = os.path.abspath(reference_rsf_path)
            
            cmd = ["java", "-jar", "mojo.jar", clust_rsf_abs, ref_rsf_abs, "-fm"]
            # print(f"   Executing MoJo: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.experiments_dir
            )
            
            if result.returncode == 0:
                # Parse MoJo-FM output (format may vary)
                output = result.stdout.strip()
                # Try to extract number
                import re
                match = re.search(r'[\d.]+', output)
                if match:
                    score = float(match.group())
                    return score
                return None
            else:
                print(f"⚠ MoJo-FM error: {result.stderr}")
                return None
        except Exception as e:
            print(f"⚠ MoJo-FM calculation failed: {e}")
            return None
    
    def calculate_metrics(self, clusters, reference_rsf_path=None):
        """Calculate all metrics for given clusters"""
        # Create temporary RSF file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rsf', delete=False) as f:
            temp_rsf = f.name
        
        try:
            # Convert clusters to RSF
            self.clusters_to_rsf(clusters, temp_rsf)
            
            # Calculate metrics
            turbomq = self.calculate_turbomq(temp_rsf)
            mojo_fm = self.calculate_mojo_fm(temp_rsf, reference_rsf_path)
            
            return {
                "turbomq": turbomq,
                "mojo_fm": mojo_fm,
                "clustering_rsf": temp_rsf  # Keep for reference
            }
        except Exception as e:
            # print(f"⚠ Metrics calculation error: {e}")
            return {"turbomq": None, "mojo_fm": None, "clustering_rsf": temp_rsf}
        finally:
            # Don't delete temp file yet - might need it
            pass
