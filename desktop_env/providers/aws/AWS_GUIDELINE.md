# ☁ Configuration of AWS

---

Welcome to the AWS VM Management documentation. Before you proceed with using the code to manage AWS services, please ensure the following variables are set correctly according to your AWS environment.

## Overview
The AWS cloud service architecture consists of a host machine that controls multiple virtual machines (each virtual machine serves as an OSWorld environment, for which we provide AMI images) for testing and potential training purposes. To prevent security breaches, we need to properly configure security groups for both the host machine and virtual machines, as well as configure appropriate subnets.

## Security Group Configuration

### Security Group for OSWorld Virtual Machines
OSWorld requires certain ports to be open, such as port 5000 for backend connections to OSWorld services, port 5910 for VNC visualization, port 9222 for Chrome control, etc. The `OSWORLD_AWS_SECURITY_GROUP_ID` variable (or legacy `AWS_SECURITY_GROUP_ID`) represents the security group configuration for virtual machines serving as OSWorld environments. Please complete the configuration and set this environment variable to the ID of the configured security group.

**⚠️ Important**: Please strictly follow the port settings below to prevent OSWorld tasks from failing due to connection issues:

#### Inbound Rules (8 rules required)

| Type | Protocol | Port Range | Source | Description |
|------|----------|------------|--------|-------------|
| SSH | TCP | 22 | 0.0.0.0/0 | SSH access |
| HTTP | TCP | 80 | 172.31.0.0/16 | HTTP traffic |
| Custom TCP | TCP | 5000 | 172.31.0.0/16 | OSWorld backend service |
| Custom TCP | TCP | 5910 | 0.0.0.0/0 | NoVNC visualization port |
| Custom TCP | TCP | 8006 | 172.31.0.0/16 | VNC service port |
| Custom TCP | TCP | 8080 | 172.31.0.0/16 | VLC service port |
| Custom TCP | TCP | 8081 | 172.31.0.0/16 | Additional service port |
| Custom TCP | TCP | 9222 | 172.31.0.0/16 | Chrome control port |

#### Outbound Rules (1 rule required)

| Type | Protocol | Port Range | Destination | Description |
|------|----------|------------|-------------|-------------|
| All traffic | All | All | 0.0.0.0/0 | Allow all outbound traffic |

### Host Machine Security Group Configuration
Configure according to your specific requirements. This project provides a monitor service that runs on port 8080 by default. You need to open this port to use this functionality.


## VPC Configuration  
To isolate the entire evaluation stack, we run both the host machine and all client virtual machines inside a dedicated VPC. The setup is straightforward:

1. Launch the host instance via the AWS console and note the **VPC ID** and **Subnet ID** shown in its network settings.  
2. Export the same **Subnet ID** as the environment variable `OSWORLD_AWS_SUBNET_ID` before starting the client code.  
   ```bash
   export OSWORLD_AWS_SUBNET_ID=subnet-xxxxxxxxxxxxxxxxx
   ```
   (Both the client and host must reside in this subnet for the evaluation to work.)
   
   > **Backward compatibility**: The old name `AWS_SUBNET_ID` still works as a fallback.


## Configuration Variables
That’s essentially all the setup you need to perform. From here on, you only have to supply a few extra details and environment variables—just make sure they’re all present in your environment.

You need to assign values to several variables crucial for the operation of these scripts on AWS:

- **`DEFAULT_REGION`**: Default AWS region where your instances will be launched.
  - Example: `"us-east-1"`
- **`IMAGE_ID_MAP`**: Dictionary mapping regions to specific AMI IDs that should be used for instance creation. Here we already set the AMI id to the official OSWorld image of Ubuntu supported by us.
  - Formatted as follows:
    ```python
    IMAGE_ID_MAP = {
        "us-east-1": "ami-0d23263edb96951d8"
        # Add other regions and corresponding AMIs
    }
    ```
- **`INSTANCE_TYPE`**: Specifies the type of EC2 instance to be launched.
  - Example: `"t3.medium"`
- **`KEY_NAME`**: Specifies the name of the key pair to be used for the instances.
  - Example: `"osworld_key"`
- **`NETWORK_INTERFACES`**: Configuration settings for network interfaces, which include subnet IDs, security group IDs, and public IP addressing.
  - Example:
    ```bash
    <!-- in .env file -->
    OSWORLD_AWS_REGION=us-east-1
    OSWORLD_AWS_SUBNET_ID=subnet-xxxx
    OSWORLD_AWS_SECURITY_GROUP_ID=sg-xxxx
    ```
    > **Backward compatibility**: The old names `AWS_REGION`, `AWS_SUBNET_ID`, `AWS_SECURITY_GROUP_ID` still work as fallbacks.

### AWS CLI Configuration
Before using these scripts, you must configure your AWS CLI with your credentials. This can be done via the following commands:

```bash
aws configure
```
This command will prompt you for:
- AWS Access Key ID
- AWS Secret Access Key
- Default region name (Optional, you can press enter)

Enter your credentials as required. This setup will allow you to interact with AWS services using the credentials provided.

### Two-Account Setup (Bedrock + OSWorld VMs)

If you use one AWS account for Bedrock (Claude LLM) and a different account for OSWorld VMs, set the `OSWORLD_AWS_*` prefixed variables for the VM account:

```bash
# Bedrock / Claude (standard AWS_* vars — used by Confucius agent)
export AWS_REGION=us-west-2
export AWS_ACCESS_KEY_ID="AKIA...BEDROCK"
export AWS_SECRET_ACCESS_KEY="...bedrock-secret..."

# OSWorld VMs (OSWORLD_AWS_* vars — used by EC2 provider)
export OSWORLD_AWS_REGION=us-east-1
export OSWORLD_AWS_ACCESS_KEY_ID="AKIA...OSWORLD"
export OSWORLD_AWS_SECRET_ACCESS_KEY="...osworld-secret..."
export OSWORLD_AWS_SUBNET_ID=subnet-xxxx
export OSWORLD_AWS_SECURITY_GROUP_ID=sg-xxxx
```

If `OSWORLD_AWS_*` vars are not set, the provider falls back to the standard `AWS_*` vars (single-account mode).

#### Optional OSWorld-specific env vars (all support `OSWORLD_` prefix with old-name fallback)

| New name (preferred) | Old name (fallback) | Description |
|---|---|---|
| `OSWORLD_AWS_SUBNET_ID` | `AWS_SUBNET_ID` | Subnet for VM instances |
| `OSWORLD_AWS_SECURITY_GROUP_ID` | `AWS_SECURITY_GROUP_ID` | Security group for VM instances |
| `OSWORLD_AWS_INSTANCE_TYPE` | `AWS_INSTANCE_TYPE` | EC2 instance type (default: `t3.large`) |
| `OSWORLD_AWS_SCHEDULER_ROLE_ARN` | `AWS_SCHEDULER_ROLE_ARN` | EventBridge Scheduler role ARN for TTL |
| `OSWORLD_AWS_SCHEDULER_ROLE_NAME` | `AWS_SCHEDULER_ROLE_NAME` | Scheduler role name (auto-derived if ARN not set) |
| `OSWORLD_AWS_AUTO_CREATE_SCHEDULER_ROLE` | `AWS_AUTO_CREATE_SCHEDULER_ROLE` | Auto-create scheduler role if missing (default: `true`) |
| `OSWORLD_DEFAULT_TTL_MINUTES` | `DEFAULT_TTL_MINUTES` | Instance TTL in minutes (default: `180`) |
| `OSWORLD_ENABLE_TTL` | `ENABLE_TTL` | Enable instance auto-termination (default: `true`) |

### Disclaimer
Use the provided scripts and configurations at your own risk. Ensure that you understand the AWS pricing model and potential costs associated with deploying instances, as using these scripts might result in charges on your AWS account.

> **Note:**  Ensure all AMI images used in `IMAGE_ID_MAP` are accessible and permissioned correctly for your AWS account, and that they are available in the specified region.
