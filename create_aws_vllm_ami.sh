#!/bin/bash
#
# DESCRIPTION:
# This script (create_aws_vllm_ami.sh) deploys an AWS GPU instance for vLLM.
# It is designed for easy switching between Blackwell (G7e) and Ada (G6e)
# architectures to bypass capacity constraints in us-east-1 and us-east-2.
#
# USAGE:
#   ./create_aws_vllm_ami.sh <region> [--on-demand]

# --- Hardware Variable Toggle ---
# Change INSTANCE_TYPE to "g6e.2xlarge" or "g7e.2xlarge" as needed.
INSTANCE_TYPE="g6e.2xlarge"
VOLUME_SIZE=120

# --- Region & Network Configuration ---
REGION="${1:-us-east-1}"
KEY_NAME="home-pc-wsl-aws-vllm"
MY_IP="71.184.241.144/32"
SG_NAME="vllm-access-sg-$REGION"
MARKET_OPTIONS='{"MarketType":"spot"}'

# Only apply your specific VPC for us-east-1
VPC_ARG=""
if [[ "$REGION" == "us-east-1" ]]; then
    VPC_ARG="--vpc-id vpc-04ae3aa1c07b66496"
fi

# --- Tier Selection ---
if [[ "$1" == "--on-demand" ]] || [[ "$2" == "--on-demand" ]]; then
    echo "!!! ON-DEMAND TIER SELECTED !!!"
    MARKET_OPTIONS=""
fi

echo "--- Deploying $INSTANCE_TYPE in $REGION ---"

# 1. Fetch Latest Deep Learning AMI
AMI_ID=$(aws ec2 describe-images --region $REGION \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.9 (Ubuntu 24.04)*" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

echo "Using AMI: $AMI_ID"

# 2. Handle Security Group
SG_ID=$(aws ec2 describe-security-groups --region $REGION \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query "SecurityGroups[0].GroupId" --output text 2>/dev/null)

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
    echo "Creating Security Group: $SG_NAME..."
    SG_ID=$(aws ec2 create-security-group --region $REGION --group-name "$SG_NAME" --description "vLLM Access" $VPC_ARG --query 'GroupId' --output text)
    aws ec2 authorize-security-group-ingress --region $REGION --group-id "$SG_ID" --protocol tcp --port 22 --cidr "$MY_IP"
fi

# 3. Execute Launch
LAUNCH_CMD="aws ec2 run-instances --region $REGION \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --block-device-mappings '[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\"}}]' \
    --network-interfaces '{\"AssociatePublicIpAddress\":true,\"DeviceIndex\":0,\"Groups\":[\"$SG_ID\"]}' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=vllm-$INSTANCE_TYPE}]' \
    --query 'Instances[0].InstanceId' --output text"

[ -n "$MARKET_OPTIONS" ] && LAUNCH_CMD="$LAUNCH_CMD --instance-market-options '$MARKET_OPTIONS'"

INSTANCE_ID=$(eval $LAUNCH_CMD 2>&1)

if [[ $INSTANCE_ID == i-* ]]; then
    echo "------------------------------------------------"
    echo "SUCCESS: $INSTANCE_ID ($INSTANCE_TYPE) is starting."
    echo "SSH: ssh -i ~/.ssh/$KEY_NAME.pem ubuntu@\$(aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
    echo "------------------------------------------------"
else
    echo "------------------------------------------------"
    echo "LAUNCH FAILED: $INSTANCE_ID"
    echo "------------------------------------------------"
fi
