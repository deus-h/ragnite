#!/usr/bin/env python3
"""
Cost estimator for cloud infrastructure usage in RAG systems.

This module provides a cost estimator for calculating and tracking the costs
associated with using cloud infrastructure in RAG systems.
"""

import datetime
from typing import Dict, List, Any, Optional, Union
import json
from .base import BaseCostEstimator


class CloudCostEstimator(BaseCostEstimator):
    """
    Cost estimator for cloud infrastructure usage.
    
    This estimator calculates and tracks the costs associated with using
    cloud infrastructure in RAG systems, including compute instances,
    storage, databases, and networking.
    
    Attributes:
        name (str): Name of the estimator.
        config (Dict[str, Any]): Configuration options for the estimator.
        price_data (Dict[str, Any]): Pricing data for cloud infrastructure.
        cloud_provider (str): Cloud provider (AWS, GCP, Azure).
    """
    
    def __init__(
        self,
        name: str,
        cloud_provider: str = "aws",
        config: Optional[Dict[str, Any]] = None,
        price_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the cloud cost estimator.
        
        Args:
            name (str): Name of the estimator.
            cloud_provider (str): Cloud provider. Valid values: 'aws', 'gcp', 'azure'.
                Defaults to 'aws'.
            config (Optional[Dict[str, Any]]): Configuration options for the estimator.
                Defaults to an empty dictionary.
            price_data (Optional[Dict[str, Any]]): Pricing data for cloud infrastructure.
                If not provided, default pricing data will be used.
        """
        self.cloud_provider = cloud_provider.lower()
        super().__init__(name=name, config=config, price_data=price_data)
    
    def _get_default_price_data(self) -> Dict[str, Any]:
        """
        Get default pricing data for cloud infrastructure.
        
        Returns:
            Dict[str, Any]: Default pricing data for cloud infrastructure.
        """
        # General pricing data as of March 2023
        # Prices are approximate and in USD
        
        # AWS pricing
        aws_pricing = {
            "compute": {
                "t4g.small": 0.0168,      # Per hour
                "t4g.medium": 0.0336,     # Per hour
                "t4g.large": 0.0672,      # Per hour
                "m6g.large": 0.0770,      # Per hour
                "m6g.xlarge": 0.154,      # Per hour
                "c6g.large": 0.0850,      # Per hour
                "c6g.xlarge": 0.170,      # Per hour
                "r6g.large": 0.1261,      # Per hour
                "r6g.xlarge": 0.2522,     # Per hour
                "g4dn.xlarge": 0.526,     # Per hour
                "g4dn.2xlarge": 0.752,    # Per hour
                "p3.2xlarge": 3.06,       # Per hour
                "p4d.24xlarge": 32.77     # Per hour
            },
            "storage": {
                "s3_standard": {
                    "storage": 0.023,     # Per GB-month
                    "put": 0.005,         # Per 1,000 requests
                    "get": 0.0004,        # Per 1,000 requests
                    "data_transfer_out": 0.09  # Per GB
                },
                "ebs_gp3": {
                    "storage": 0.08,      # Per GB-month
                    "iops": 0.005,        # Per IOPS-month (over 3,000)
                    "throughput": 0.04    # Per MB/s-month (over 125)
                }
            },
            "database": {
                "rds_postgres": {
                    "db.t4g.micro": 0.016,    # Per hour
                    "db.t4g.small": 0.032,    # Per hour
                    "db.t4g.medium": 0.064,   # Per hour
                    "db.m6g.large": 0.155,    # Per hour
                    "storage": 0.115          # Per GB-month
                },
                "dynamodb": {
                    "write": 1.25,        # Per million write request units
                    "read": 0.25,         # Per million read request units
                    "storage": 0.25       # Per GB-month
                },
                "opensearch": {
                    "t3.small.search": 0.036,   # Per hour
                    "m6g.large.search": 0.156,  # Per hour
                    "c6g.large.search": 0.156,  # Per hour
                    "r6g.large.search": 0.186,  # Per hour
                    "storage": 0.135            # Per GB-month
                }
            },
            "networking": {
                "data_transfer_out": {
                    "first_10TB": 0.09,   # Per GB
                    "next_40TB": 0.085,   # Per GB
                    "next_100TB": 0.07,   # Per GB
                    "over_150TB": 0.05    # Per GB
                },
                "elastic_ip": 0.005,      # Per unused IP per hour
                "load_balancer": {
                    "application": 0.0225,  # Per hour
                    "network": 0.0225       # Per hour
                }
            }
        }
        
        # GCP pricing
        gcp_pricing = {
            "compute": {
                "e2-small": 0.0209,       # Per hour
                "e2-medium": 0.0350,      # Per hour
                "e2-standard-2": 0.0699,  # Per hour
                "e2-standard-4": 0.1399,  # Per hour
                "n2-standard-2": 0.0971,  # Per hour
                "n2-standard-4": 0.1942,  # Per hour
                "n2-standard-8": 0.3885,  # Per hour
                "t4-standard": 0.35,      # Per hour
                "a100-standard": 3.6314   # Per hour
            },
            "storage": {
                "standard_storage": {
                    "storage": 0.02,      # Per GB-month
                    "class_a_operations": 0.005,  # Per 10,000 operations
                    "class_b_operations": 0.0004  # Per 10,000 operations
                },
                "persistent_disk": {
                    "standard": 0.04,     # Per GB-month
                    "ssd": 0.17           # Per GB-month
                }
            },
            "database": {
                "cloud_sql_postgres": {
                    "db-f1-micro": 0.0150,  # Per hour
                    "db-g1-small": 0.0350,  # Per hour
                    "db-custom-1-3840": 0.0755,  # Per hour
                    "db-custom-2-7680": 0.1511,  # Per hour
                    "storage": 0.17         # Per GB-month
                },
                "firestore": {
                    "document_reads": 0.06,   # Per 100,000 reads
                    "document_writes": 0.18,  # Per 100,000 writes
                    "document_deletes": 0.02,  # Per 100,000 deletes
                    "storage": 0.18           # Per GB-month
                }
            },
            "networking": {
                "data_transfer_out": {
                    "in_region": 0.00,    # Per GB
                    "same_continent": 0.02,  # Per GB
                    "different_continent": 0.12  # Per GB
                },
                "load_balancer": {
                    "forwarding_rule": 0.025,  # Per hour
                    "data_processed": 0.008    # Per GB
                }
            }
        }
        
        # Azure pricing
        azure_pricing = {
            "compute": {
                "b1s": 0.0124,           # Per hour
                "b2s": 0.0496,           # Per hour
                "d2s_v3": 0.0998,        # Per hour
                "d4s_v3": 0.1996,        # Per hour
                "e2s_v3": 0.098,         # Per hour
                "e4s_v3": 0.196,         # Per hour
                "nc6s_v3": 3.06,         # Per hour
                "nc24s_v3": 12.24        # Per hour
            },
            "storage": {
                "blob_storage": {
                    "hot_storage": 0.0184,  # Per GB-month
                    "cool_storage": 0.01,   # Per GB-month
                    "hot_write": 0.055,     # Per 10,000 operations
                    "hot_read": 0.0044,     # Per 10,000 operations
                    "cool_write": 0.10,     # Per 10,000 operations
                    "cool_read": 0.01       # Per 10,000 operations
                },
                "managed_disk": {
                    "standard_hdd": 0.05,   # Per GB-month
                    "standard_ssd": 0.08,   # Per GB-month
                    "premium_ssd": 0.17     # Per GB-month
                }
            },
            "database": {
                "postgres": {
                    "basic_1": 0.036,     # Per hour
                    "basic_2": 0.071,     # Per hour
                    "gp_gen5_2": 0.258,   # Per hour
                    "gp_gen5_4": 0.516,   # Per hour
                    "storage": 0.115      # Per GB-month
                },
                "cosmos_db": {
                    "provisioned": 0.008,  # Per RU/hour (100 RUs)
                    "serverless": 0.28,    # Per million RUs
                    "storage": 0.25        # Per GB-month
                }
            },
            "networking": {
                "data_transfer_out": {
                    "in_region": 0.00,    # Per GB
                    "intra_continent": 0.05,  # Per GB
                    "inter_continent": 0.087  # Per GB
                },
                "load_balancer": {
                    "standard": 0.0225,   # Per hour
                    "data_processed": 0.005  # Per GB
                }
            }
        }
        
        pricing = {
            "aws": aws_pricing,
            "gcp": gcp_pricing,
            "azure": azure_pricing
        }
        
        # Return pricing data for the specified cloud provider
        return pricing.get(self.cloud_provider, aws_pricing)
    
    def estimate_cost(
        self, 
        usage: Dict[str, Any],
        time_period: Optional[Dict[str, datetime.datetime]] = None
    ) -> Dict[str, Any]:
        """
        Estimate the cost based on cloud infrastructure usage.
        
        Args:
            usage (Dict[str, Any]): Usage data for cloud infrastructure.
                Expected format:
                {
                    "compute": {
                        "instance_type": str,
                        "hours": float
                    },
                    "storage": {
                        "type": str,  # "s3_standard", "ebs_gp3", etc.
                        "storage_gb": float,
                        "requests": {
                            "put": int,
                            "get": int
                        },
                        "data_transfer_out_gb": float,
                        "iops": int,  # Only for EBS
                        "throughput_mbps": int  # Only for EBS
                    },
                    "database": {
                        "type": str,  # "rds_postgres", "dynamodb", etc.
                        "instance_type": str,  # Only for RDS
                        "storage_gb": float,
                        "write_units": int,  # Only for DynamoDB
                        "read_units": int,  # Only for DynamoDB
                        "hours": float  # Only for RDS and OpenSearch
                    },
                    "networking": {
                        "data_transfer_out_gb": float,
                        "elastic_ip_hours": float,  # Only for AWS
                        "load_balancer": {
                            "type": str,  # "application", "network", etc.
                            "hours": float
                        }
                    }
                }
            time_period (Optional[Dict[str, datetime.datetime]]): Time period for the cost estimation.
                Should contain 'start' and 'end' keys with datetime values.
                If not provided, the entire usage history will be considered.
        
        Returns:
            Dict[str, Any]: Cost estimation result with detailed breakdown.
        """
        total_cost = 0.0
        items = []
        
        # Compute cost
        if "compute" in usage:
            compute_usage = usage["compute"]
            instance_type = compute_usage.get("instance_type", "")
            hours = compute_usage.get("hours", 0)
            
            # Get instance type pricing
            instance_price = self.price_data["compute"].get(instance_type, 0)
            
            # Calculate cost
            compute_cost = instance_price * hours
            total_cost += compute_cost
            
            # Add to items
            items.append({
                "category": "Compute",
                "instance_type": instance_type,
                "hours": hours,
                "hourly_rate": instance_price,
                "cost": round(compute_cost, 4)
            })
        
        # Storage cost
        if "storage" in usage:
            storage_usage = usage["storage"]
            storage_type = storage_usage.get("type", "")
            storage_gb = storage_usage.get("storage_gb", 0)
            requests = storage_usage.get("requests", {})
            data_transfer_out_gb = storage_usage.get("data_transfer_out_gb", 0)
            iops = storage_usage.get("iops", 0)
            throughput_mbps = storage_usage.get("throughput_mbps", 0)
            
            storage_cost = 0.0
            
            # Calculate based on storage type
            if self.cloud_provider == "aws":
                if storage_type == "s3_standard":
                    # S3 Standard storage cost
                    storage_cost += storage_gb * self.price_data["storage"]["s3_standard"]["storage"]
                    
                    # S3 requests cost
                    put_requests = requests.get("put", 0)
                    get_requests = requests.get("get", 0)
                    storage_cost += (put_requests / 1000) * self.price_data["storage"]["s3_standard"]["put"]
                    storage_cost += (get_requests / 1000) * self.price_data["storage"]["s3_standard"]["get"]
                    
                    # S3 data transfer cost
                    storage_cost += data_transfer_out_gb * self.price_data["storage"]["s3_standard"]["data_transfer_out"]
                
                elif storage_type == "ebs_gp3":
                    # EBS GP3 storage cost
                    storage_cost += storage_gb * self.price_data["storage"]["ebs_gp3"]["storage"]
                    
                    # EBS IOPS cost (over 3,000 IOPS)
                    if iops > 3000:
                        storage_cost += (iops - 3000) * self.price_data["storage"]["ebs_gp3"]["iops"]
                    
                    # EBS throughput cost (over 125 MB/s)
                    if throughput_mbps > 125:
                        storage_cost += (throughput_mbps - 125) * self.price_data["storage"]["ebs_gp3"]["throughput"]
            
            elif self.cloud_provider == "gcp":
                if storage_type == "standard_storage":
                    # GCS Standard storage cost
                    storage_cost += storage_gb * self.price_data["storage"]["standard_storage"]["storage"]
                    
                    # GCS requests cost
                    class_a_operations = requests.get("put", 0)
                    class_b_operations = requests.get("get", 0)
                    storage_cost += (class_a_operations / 10000) * self.price_data["storage"]["standard_storage"]["class_a_operations"]
                    storage_cost += (class_b_operations / 10000) * self.price_data["storage"]["standard_storage"]["class_b_operations"]
                
                elif storage_type in ["persistent_disk", "ssd"]:
                    # Persistent disk storage cost
                    storage_cost += storage_gb * self.price_data["storage"]["persistent_disk"][storage_type]
            
            elif self.cloud_provider == "azure":
                if storage_type in ["hot_storage", "cool_storage"]:
                    # Blob storage cost
                    storage_cost += storage_gb * self.price_data["storage"]["blob_storage"][storage_type]
                    
                    # Blob storage operations cost
                    put_requests = requests.get("put", 0)
                    get_requests = requests.get("get", 0)
                    storage_cost += (put_requests / 10000) * self.price_data["storage"]["blob_storage"][storage_type + "_write"]
                    storage_cost += (get_requests / 10000) * self.price_data["storage"]["blob_storage"][storage_type + "_read"]
                
                elif storage_type in ["standard_hdd", "standard_ssd", "premium_ssd"]:
                    # Managed disk storage cost
                    storage_cost += storage_gb * self.price_data["storage"]["managed_disk"][storage_type]
            
            total_cost += storage_cost
            
            # Add to items
            items.append({
                "category": "Storage",
                "type": storage_type,
                "storage_gb": storage_gb,
                "requests": requests,
                "data_transfer_out_gb": data_transfer_out_gb,
                "iops": iops,
                "throughput_mbps": throughput_mbps,
                "cost": round(storage_cost, 4)
            })
        
        # Database cost
        if "database" in usage:
            database_usage = usage["database"]
            db_type = database_usage.get("type", "")
            instance_type = database_usage.get("instance_type", "")
            storage_gb = database_usage.get("storage_gb", 0)
            write_units = database_usage.get("write_units", 0)
            read_units = database_usage.get("read_units", 0)
            hours = database_usage.get("hours", 0)
            
            database_cost = 0.0
            
            # Calculate based on database type
            if self.cloud_provider == "aws":
                if db_type == "rds_postgres":
                    # RDS instance cost
                    instance_price = self.price_data["database"]["rds_postgres"].get(instance_type, 0)
                    database_cost += instance_price * hours
                    
                    # RDS storage cost
                    database_cost += storage_gb * self.price_data["database"]["rds_postgres"]["storage"]
                
                elif db_type == "dynamodb":
                    # DynamoDB write capacity cost
                    database_cost += (write_units / 1000000) * self.price_data["database"]["dynamodb"]["write"]
                    
                    # DynamoDB read capacity cost
                    database_cost += (read_units / 1000000) * self.price_data["database"]["dynamodb"]["read"]
                    
                    # DynamoDB storage cost
                    database_cost += storage_gb * self.price_data["database"]["dynamodb"]["storage"]
                
                elif db_type == "opensearch":
                    # OpenSearch instance cost
                    instance_price = self.price_data["database"]["opensearch"].get(instance_type, 0)
                    database_cost += instance_price * hours
                    
                    # OpenSearch storage cost
                    database_cost += storage_gb * self.price_data["database"]["opensearch"]["storage"]
            
            elif self.cloud_provider == "gcp":
                if db_type == "cloud_sql_postgres":
                    # Cloud SQL instance cost
                    instance_price = self.price_data["database"]["cloud_sql_postgres"].get(instance_type, 0)
                    database_cost += instance_price * hours
                    
                    # Cloud SQL storage cost
                    database_cost += storage_gb * self.price_data["database"]["cloud_sql_postgres"]["storage"]
                
                elif db_type == "firestore":
                    # Firestore operations cost
                    database_cost += (read_units / 100000) * self.price_data["database"]["firestore"]["document_reads"]
                    database_cost += (write_units / 100000) * self.price_data["database"]["firestore"]["document_writes"]
                    
                    # Firestore storage cost
                    database_cost += storage_gb * self.price_data["database"]["firestore"]["storage"]
            
            elif self.cloud_provider == "azure":
                if db_type == "postgres":
                    # Azure Database for PostgreSQL instance cost
                    instance_price = self.price_data["database"]["postgres"].get(instance_type, 0)
                    database_cost += instance_price * hours
                    
                    # Azure Database for PostgreSQL storage cost
                    database_cost += storage_gb * self.price_data["database"]["postgres"]["storage"]
                
                elif db_type == "cosmos_db":
                    # Cosmos DB provisioned throughput cost
                    database_cost += (read_units / 100) * self.price_data["database"]["cosmos_db"]["provisioned"] * hours
                    
                    # Cosmos DB storage cost
                    database_cost += storage_gb * self.price_data["database"]["cosmos_db"]["storage"]
            
            total_cost += database_cost
            
            # Add to items
            items.append({
                "category": "Database",
                "type": db_type,
                "instance_type": instance_type,
                "storage_gb": storage_gb,
                "write_units": write_units,
                "read_units": read_units,
                "hours": hours,
                "cost": round(database_cost, 4)
            })
        
        # Networking cost
        if "networking" in usage:
            networking_usage = usage["networking"]
            data_transfer_out_gb = networking_usage.get("data_transfer_out_gb", 0)
            elastic_ip_hours = networking_usage.get("elastic_ip_hours", 0)
            load_balancer = networking_usage.get("load_balancer", {})
            lb_type = load_balancer.get("type", "")
            lb_hours = load_balancer.get("hours", 0)
            
            networking_cost = 0.0
            
            # Calculate data transfer cost
            if self.cloud_provider == "aws":
                # AWS data transfer cost (simplified)
                if data_transfer_out_gb <= 10000:
                    networking_cost += data_transfer_out_gb * self.price_data["networking"]["data_transfer_out"]["first_10TB"]
                else:
                    networking_cost += 10000 * self.price_data["networking"]["data_transfer_out"]["first_10TB"]
                    remaining_gb = data_transfer_out_gb - 10000
                    
                    if remaining_gb <= 40000:
                        networking_cost += remaining_gb * self.price_data["networking"]["data_transfer_out"]["next_40TB"]
                    else:
                        networking_cost += 40000 * self.price_data["networking"]["data_transfer_out"]["next_40TB"]
                        remaining_gb -= 40000
                        
                        if remaining_gb <= 100000:
                            networking_cost += remaining_gb * self.price_data["networking"]["data_transfer_out"]["next_100TB"]
                        else:
                            networking_cost += 100000 * self.price_data["networking"]["data_transfer_out"]["next_100TB"]
                            remaining_gb -= 100000
                            networking_cost += remaining_gb * self.price_data["networking"]["data_transfer_out"]["over_150TB"]
                
                # AWS elastic IP cost
                networking_cost += elastic_ip_hours * self.price_data["networking"]["elastic_ip"]
                
                # AWS load balancer cost
                if lb_type in ["application", "network"]:
                    networking_cost += lb_hours * self.price_data["networking"]["load_balancer"][lb_type]
            
            elif self.cloud_provider == "gcp":
                # GCP data transfer cost (simplified)
                networking_cost += data_transfer_out_gb * self.price_data["networking"]["data_transfer_out"]["different_continent"]
                
                # GCP load balancer cost
                if lb_type == "forwarding_rule":
                    networking_cost += lb_hours * self.price_data["networking"]["load_balancer"]["forwarding_rule"]
            
            elif self.cloud_provider == "azure":
                # Azure data transfer cost (simplified)
                networking_cost += data_transfer_out_gb * self.price_data["networking"]["data_transfer_out"]["inter_continent"]
                
                # Azure load balancer cost
                if lb_type == "standard":
                    networking_cost += lb_hours * self.price_data["networking"]["load_balancer"]["standard"]
            
            total_cost += networking_cost
            
            # Add to items
            items.append({
                "category": "Networking",
                "data_transfer_out_gb": data_transfer_out_gb,
                "elastic_ip_hours": elastic_ip_hours,
                "load_balancer_type": lb_type,
                "load_balancer_hours": lb_hours,
                "cost": round(networking_cost, 4)
            })
        
        # Prepare result
        time_info = {}
        if time_period:
            time_info = {
                "start_time": time_period.get("start", "").isoformat() if time_period.get("start") else None,
                "end_time": time_period.get("end", "").isoformat() if time_period.get("end") else None
            }
        
        result = {
            "provider": self.cloud_provider.upper(),
            "time_period": time_info,
            "total_cost": round(total_cost, 4),
            "currency": "USD",
            "items": items
        }
        
        return result
    
    def _aggregate_usage(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate cloud infrastructure usage data from multiple records.
        
        Args:
            records (List[Dict[str, Any]]): List of usage records.
        
        Returns:
            Dict[str, Any]: Aggregated usage data.
        """
        # Initialize result with empty structures
        result = {
            "compute": {},
            "storage": {},
            "database": {},
            "networking": {}
        }
        
        # Track instance types and their hours
        compute_instances = {}
        database_instances = {}
        
        for record in records:
            # Process compute usage
            if "compute" in record:
                compute = record["compute"]
                instance_type = compute.get("instance_type", "")
                hours = compute.get("hours", 0)
                
                if instance_type:
                    if instance_type not in compute_instances:
                        compute_instances[instance_type] = 0
                    compute_instances[instance_type] += hours
            
            # Process storage usage
            if "storage" in record:
                storage = record["storage"]
                storage_type = storage.get("type", "")
                storage_gb = storage.get("storage_gb", 0)
                requests = storage.get("requests", {})
                data_transfer_out_gb = storage.get("data_transfer_out_gb", 0)
                iops = storage.get("iops", 0)
                throughput_mbps = storage.get("throughput_mbps", 0)
                
                if storage_type not in result["storage"]:
                    result["storage"][storage_type] = {
                        "storage_gb": 0,
                        "requests": {
                            "put": 0,
                            "get": 0
                        },
                        "data_transfer_out_gb": 0,
                        "iops": 0,
                        "throughput_mbps": 0
                    }
                
                result["storage"][storage_type]["storage_gb"] += storage_gb
                result["storage"][storage_type]["requests"]["put"] += requests.get("put", 0)
                result["storage"][storage_type]["requests"]["get"] += requests.get("get", 0)
                result["storage"][storage_type]["data_transfer_out_gb"] += data_transfer_out_gb
                result["storage"][storage_type]["iops"] += iops
                result["storage"][storage_type]["throughput_mbps"] += throughput_mbps
            
            # Process database usage
            if "database" in record:
                database = record["database"]
                db_type = database.get("type", "")
                instance_type = database.get("instance_type", "")
                storage_gb = database.get("storage_gb", 0)
                write_units = database.get("write_units", 0)
                read_units = database.get("read_units", 0)
                hours = database.get("hours", 0)
                
                if db_type not in result["database"]:
                    result["database"][db_type] = {
                        "storage_gb": 0,
                        "write_units": 0,
                        "read_units": 0
                    }
                
                result["database"][db_type]["storage_gb"] += storage_gb
                result["database"][db_type]["write_units"] += write_units
                result["database"][db_type]["read_units"] += read_units
                
                if instance_type:
                    key = f"{db_type}:{instance_type}"
                    if key not in database_instances:
                        database_instances[key] = 0
                    database_instances[key] += hours
            
            # Process networking usage
            if "networking" in record:
                networking = record["networking"]
                data_transfer_out_gb = networking.get("data_transfer_out_gb", 0)
                elastic_ip_hours = networking.get("elastic_ip_hours", 0)
                load_balancer = networking.get("load_balancer", {})
                
                if "data_transfer_out_gb" not in result["networking"]:
                    result["networking"]["data_transfer_out_gb"] = 0
                result["networking"]["data_transfer_out_gb"] += data_transfer_out_gb
                
                if "elastic_ip_hours" not in result["networking"]:
                    result["networking"]["elastic_ip_hours"] = 0
                result["networking"]["elastic_ip_hours"] += elastic_ip_hours
                
                if load_balancer:
                    lb_type = load_balancer.get("type", "")
                    lb_hours = load_balancer.get("hours", 0)
                    
                    if "load_balancer" not in result["networking"]:
                        result["networking"]["load_balancer"] = {}
                    
                    if lb_type not in result["networking"]["load_balancer"]:
                        result["networking"]["load_balancer"][lb_type] = 0
                    
                    result["networking"]["load_balancer"][lb_type] += lb_hours
        
        # Add compute instances to result
        for instance_type, hours in compute_instances.items():
            result["compute"][instance_type] = {"hours": hours}
        
        # Add database instances to result
        for key, hours in database_instances.items():
            db_type, instance_type = key.split(":", 1)
            if "instance_hours" not in result["database"][db_type]:
                result["database"][db_type]["instance_hours"] = {}
            result["database"][db_type]["instance_hours"][instance_type] = hours
        
        return result 